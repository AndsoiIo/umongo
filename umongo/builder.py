"""Builder module

A builder connect a :class:`umongo.document.Template` with a
:class:`umongo.instance.BaseInstance` by generating an
:class:`umongo.document.Implementation`.
"""
import re
import inspect
from copy import copy

import marshmallow as ma

from .template import Template, Implementation
from .data_proxy import data_proxy_factory
from .document import DocumentTemplate, DocumentOpts, DocumentImplementation
from .embedded_document import (
    EmbeddedDocumentTemplate, EmbeddedDocumentOpts, EmbeddedDocumentImplementation)
from .exceptions import DocumentDefinitionError, NotRegisteredDocumentError
from .schema import Schema
from .indexes import parse_index
from . import fields


def camel_to_snake(name):
    tmp_str = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', tmp_str).lower()


def _is_child(template, base_tmpl_cls):
    """Return true if the (embedded) document has a concrete parent"""
    return any(
        b for b in template.__bases__
        if issubclass(b, base_tmpl_cls) and
        b is not base_tmpl_cls and
        ('Meta' not in b.__dict__ or not getattr(b.Meta, 'abstract', False))
    )


def _on_need_add_id_field(bases, fields_dict):
    """
    If the given fields make no reference to `_id`, add an `id` field
    (type ObjectId, dump_only=True, attribute=`_id`) to handle it
    """

    def find_id_field(fields_dict):
        for name, field in fields_dict.items():
            # Skip fake fields present in schema (e.g. `post_load` decorated function)
            if not isinstance(field, ma.fields.Field):
                continue
            if (name == '_id' and not field.attribute) or field.attribute == '_id':
                return name
        return None

    # Search among parents for the id field
    for base in bases:
        schema = base()
        name = find_id_field(schema.fields)
        if name is not None:
            return name

    # Search among our own fields
    name = find_id_field(fields_dict)
    if name is not None:
        return name

    # No id field found, add a default one
    fields_dict['id'] = fields.ObjectIdField(attribute='_id', dump_only=True)
    return 'id'


def _add_child_field(name, fields_dict):
    fields_dict['cls'] = fields.StringField(attribute='_cls', default=name, dump_only=True)


def _collect_schema_attrs(template):
    """
    Split dict between schema fields and non-fields elements and retrieve
    marshmallow tags if any.
    """
    schema_fields = {}
    schema_non_fields = {}
    nmspc = {}
    for key, item in template.__dict__.items():
        if hasattr(item, '__marshmallow_hook__'):
            # Decorated special functions (e.g. `post_load`)
            schema_non_fields[key] = item
        elif isinstance(item, ma.fields.Field):
            # Given the fields provided by the template are going to be
            # customized in the implementation, we copy them to avoid
            # overwriting if two implementations are created
            schema_fields[key] = copy(item)
        else:
            nmspc[key] = item
    return nmspc, schema_fields, schema_non_fields


def _collect_indexes(meta, schema_nmspc, bases, is_child):
    """
    Retrieve all indexes (custom defined in meta class, by inheritances
    and unique attribut in fields)
    """
    indexes = []

    # First collect parent indexes (including inherited field's unique indexes)
    for base in bases:
        if issubclass(base, DocumentImplementation):
            indexes += base.opts.indexes

    # Then get our own custom indexes
    if is_child:
        custom_indexes = [
            parse_index(x, base_compound_field='_cls')
            for x in getattr(meta, 'indexes', ())
        ]
    else:
        custom_indexes = [parse_index(x) for x in getattr(meta, 'indexes', ())]
    indexes += custom_indexes

    if is_child:
        indexes.append(parse_index('_cls'))

    # Finally parse our own fields (i.e. not inherited) for unique indexes
    def parse_field(mongo_path, path, field):
        if field.unique:
            index = {'unique': True, 'key': [mongo_path]}
            if not field.required or field.allow_none:
                index['sparse'] = True
            if is_child:
                index['key'].append('_cls')
            indexes.append(parse_index(index))

    for name, field in schema_nmspc.items():
        parse_field(name or field.attribute, name, field)
        if hasattr(field, 'map_to_field'):
            field.map_to_field(name or field.attribute, name, parse_field)

    return indexes


class BaseBuilder:
    """
    A builder connect a :class:`umongo.document.Template` with a
    :class:`umongo.instance.BaseInstance` by generating an
    :class:`umongo.document.Implementation`.

    .. note:: This class should not be used directly, it should be inherited by
              concrete implementations such as :class:`umongo.frameworks.pymongo.PyMongoBuilder`
    """

    BASE_DOCUMENT_CLS = None

    def __init__(self, instance):
        assert self.BASE_DOCUMENT_CLS
        self.instance = instance
        self._templates_lookup = {
            DocumentTemplate: self.BASE_DOCUMENT_CLS,
            EmbeddedDocumentTemplate: EmbeddedDocumentImplementation
        }

    def _convert_bases(self, bases):
        "Replace template parents by their implementation inside this instance"
        converted_bases = []
        for base in bases:
            assert not issubclass(base, Implementation), \
                'Document cannot inherit of implementations'
            if issubclass(base, Template):
                if base not in self._templates_lookup:
                    raise NotRegisteredDocumentError('Unknown document `%r`' % base)
                converted_bases.append(self._templates_lookup[base])
            else:
                converted_bases.append(base)
        return tuple(converted_bases)

    def _patch_field(self, field):
        # Recursively set the `instance` attribute to all fields
        field.instance = self.instance
        if isinstance(field, fields.ListField):
            self._patch_field(field.inner)
        elif isinstance(field, fields.DictField):
            if field.key_field:
                self._patch_field(field.key_field)
            if field.value_field:
                self._patch_field(field.value_field)
        elif isinstance(field, fields.EmbeddedField):
            for embedded_field in field.schema.fields.values():
                self._patch_field(embedded_field)

    def _build_schema(self, template, schema_bases, schema_fields, schema_non_fields):
        # Recursively set the `instance` attribute to all fields
        for field in schema_fields.values():
            self._patch_field(field)

        # Finally build the schema class
        schema_nmspc = {}
        schema_nmspc.update(schema_fields)
        schema_nmspc.update(schema_non_fields)
        schema_nmspc['MA_BASE_SCHEMA_CLS'] = template.MA_BASE_SCHEMA_CLS
        return type('%sSchema' % template.__name__, schema_bases, schema_nmspc)

    def _build_document_opts(self, template, bases, is_child):
        embedded = not issubclass(template, DocumentTemplate)
        base_opts_cls = EmbeddedDocumentOpts if embedded else DocumentOpts
        base_impl_cls = EmbeddedDocumentImplementation if embedded else DocumentImplementation
        kwargs = {}
        meta = template.__dict__.get('Meta')
        collection_name = getattr(meta, 'collection_name', None)
        kwargs['instance'] = self.instance
        kwargs['template'] = template
        kwargs['abstract'] = getattr(meta, 'abstract', False)
        kwargs['is_child'] = is_child
        kwargs['strict'] = getattr(meta, 'strict', True)

        # Handle option inheritance and integrity checks
        for base in bases:
            if not issubclass(base, base_impl_cls):
                continue
            popts = base.opts
            if kwargs['abstract'] and not popts.abstract:
                raise DocumentDefinitionError(
                    "Abstract document should have all it parents abstract")
            if not embedded and popts.collection_name:
                if collection_name:
                    raise DocumentDefinitionError(
                        "Cannot redefine collection_name in a child, use abstract instead")
                collection_name = popts.collection_name

        if not embedded:
            if collection_name:
                if kwargs['abstract']:
                    raise DocumentDefinitionError(
                        'Abstract document cannot define collection_name')
            elif not kwargs['abstract']:
                # Determine the collection name from the class name
                collection_name = camel_to_snake(template.__name__)
            kwargs['collection_name'] = collection_name

        return base_opts_cls(**kwargs)

    def build_from_template(self, template):
        """
        Generate a :class:`umongo.document.DocumentImplementation` for this
        instance from the given :class:`umongo.document.DocumentTemplate`.
        """
        embedded = not issubclass(template, DocumentTemplate)
        base_tmpl_cls = EmbeddedDocumentTemplate if embedded else DocumentTemplate
        base_impl_cls = EmbeddedDocumentImplementation if embedded else DocumentImplementation
        is_child = _is_child(template, base_tmpl_cls)
        name = template.__name__
        bases = self._convert_bases(template.__bases__)
        opts = self._build_document_opts(template, bases, is_child)
        nmspc, schema_fields, schema_non_fields = _collect_schema_attrs(template)
        nmspc['opts'] = opts

        # Create schema by retrieving inherited schema classes
        schema_bases = tuple(
            base.Schema for base in bases
            if issubclass(base, Implementation) and hasattr(base, 'Schema')
        )
        if not schema_bases:
            schema_bases = (Schema, )
        if not embedded:
            nmspc['pk_field'] = _on_need_add_id_field(schema_bases, schema_fields)

        if is_child:
            _add_child_field(name, schema_fields)
        schema_cls = self._build_schema(template, schema_bases, schema_fields, schema_non_fields)
        nmspc['Schema'] = schema_cls
        schema = schema_cls()
        nmspc['schema'] = schema
        nmspc['DataProxy'] = data_proxy_factory(name, schema, strict=opts.strict)

        if not embedded:
            # _build_document_opts cannot determine the indexes given we need to
            # visit the document's fields which weren't defined at this time
            opts.indexes = _collect_indexes(nmspc.get('Meta'), schema.fields, bases, is_child)

        implementation = type(name, bases, nmspc)
        self._templates_lookup[template] = implementation
        # Notify the parent & grand parents of the newborn !
        for base in bases:
            for parent in inspect.getmro(base):
                if issubclass(parent, base_impl_cls) and parent is not base_impl_cls:
                    parent.opts.offspring.add(implementation)
        return implementation
