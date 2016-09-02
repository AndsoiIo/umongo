from marshmallow import (Schema as MaSchema, fields as ma_fields,
                         validate as ma_validate, missing, validates_schema)

from .exceptions import ValidationError
from .i18n import gettext as _


__all__ = ('BaseSchema', 'BaseField', 'BaseValidator', 'BaseDataObject')


def _check_unknown_fields(self, data, original_data):
    loadable_fields = [k for k, v in self.fields.items() if not v.dump_only]
    for key in original_data:
        if key not in loadable_fields:
            raise ValidationError(_('Unknown field name {field}.').format(field=key))


class BaseSchema(MaSchema):
    """
    All schema used in umongo should inherit from this base schema
    """

    __check_unknown_fields = validates_schema(pass_original=True)(_check_unknown_fields)

    def map_to_field(self, func):
        """
        Apply a function to every field in the schema

        >>> def func(mongo_path, path, field):
        ...     pass
        """
        for name, field in self.fields.items():
            mongo_path = field.attribute or name
            func(mongo_path, name, field)
            if hasattr(field, 'map_to_field'):
                field.map_to_field(mongo_path, name, func)

    def as_marshmallow_schema(self, params=None, base_schema_cls=MaSchema,
                              check_unknown_fields=True, mongo_world=False):
        """
        Return a pure-marshmallow version of this schema class.

        :param params: Per-field dict to pass parameters to their field creation.
        :param base_schema_cls: Class the schema will inherit from (
            default: :class:`marshmallow.Schema`).
        :param check_unknown_fields: Unknown fields are considered as errors (default: True).
        :param mongo_world: If True the schema will work against the mongo world
            instead of the OO world (default: False).
        """
        params = params or {}
        nmspc = {name: field.as_marshmallow_field(params=params.get(name), mongo_world=mongo_world)
                 for name, field in self.fields.items()}
        if check_unknown_fields:
            nmspc['__check_unknown_fields'] = validates_schema(
                pass_original=True)(_check_unknown_fields)
        return type('Marshmallow%s' % type(self).__name__, (base_schema_cls, ), nmspc)


class I18nErrorDict(dict):
    def __getitem__(self, name):
        raw_msg = dict.__getitem__(self, name)
        return _(raw_msg)


class BaseField(ma_fields.Field):
    """
    All fields used in umongo should inherit from this base field.

    Enabled flags                 | resulting index
    ------------------------------+----------------
    <no flags>                    |
    allow_none                    |
    required                      |
    required, allow_none          |
    required, unique, allow_none  | unique
    unique                        | unique, sparse
    unique, required              | unique
    unique, allow_none            | unique, sparse

    Note: Even with allow_none flag, the unique flag will refuse duplicated
    `null` value (consider unsetting the field with `del` instead)
    """

    default_error_messages = {
        'unique': 'Field value must be unique.',
        'unique_compound': 'Values of fields {fields} must be unique together.'
    }

    def __init__(self, *args, io_validate=None, unique=False, instance=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Overwrite error_messages to handle i18n translation
        self.error_messages = I18nErrorDict(self.error_messages)
        self.io_validate = io_validate
        self.unique = unique
        self.instance = instance

    def __repr__(self):
        return ('<fields.{ClassName}(default={self.default!r}, '
                'attribute={self.attribute!r}, '
                'validate={self.validate}, required={self.required}, '
                'load_only={self.load_only}, dump_only={self.dump_only}, '
                'missing={self.missing}, allow_none={self.allow_none}, '
                'error_messages={self.error_messages}, '
                'io_validate={self.io_validate}, unique={self.unique}, '
                'instance={self.instance})>'
                .format(ClassName=self.__class__.__name__, self=self))

    def serialize(self, attr, obj, accessor=None):
        return super().serialize(attr, obj, accessor=accessor)

    def deserialize(self, value, attr=None, data=None):
        return super().deserialize(value, attr=attr, data=data)

    def serialize_to_mongo(self, obj):
        if obj is missing:
            return missing
        return self._serialize_to_mongo(obj)

    # def serialize_to_mongo_update(self, path, obj):
    #     return self._serialize_to_mongo(attr, obj=obj, update=update)

    def deserialize_from_mongo(self, value):
        return self._deserialize_from_mongo(value)

    def _serialize_to_mongo(self, obj):
        return obj

    # def _serialize_to_mongo_update(self, ):
    #     if isinstance(obj, BaseDataObject):
    #         return obj.to_mongo(attr=attr, update=update)
    #     elif update:
    #         return {attr: obj}
    #     else:
    #         return obj

    def _deserialize_from_mongo(self, value):
        return value

    # # Hat tip to django-rest-framework.
    # def fail(self, key, **kwargs):
    #     """A helper method that simply raises a `ValidationError`.
    #     """
    #     from .exceptions import ValidationError
    #     try:
    #         error = self.error_messages[key]
    #         msg = error if not callable(error) else error(self.context)
    #     except KeyError:
    #         class_name = self.__class__.__name__
    #         from marshmallow.fields import MISSING_ERROR_MESSAGE
    #         msg = MISSING_ERROR_MESSAGE.format(class_name=class_name, key=key)
    #         raise AssertionError(msg)
    #     if isinstance(msg, str):
    #         msg = msg.format(**kwargs)
    #     raise ValidationError(msg)

    def translate_query(self, key, query):
        return {self.attribute or key: query}

    def _extract_marshmallow_field_params(self, mongo_world):
        params = {field: getattr(self, field)
                  for field in ('default', 'load_from', 'validate',
                                'required', 'allow_none', 'load_only',
                                'dump_only', 'missing', 'error_messages')}
        if mongo_world and self.attribute:
            params['attribute'] = self.attribute
        return params

    def as_marshmallow_field(self, params=None, mongo_world=False):
        """
        Return a pure-marshmallow version of this field.

        :param params: Additional parameters passed to the mashmallow field
            class constructor.
        :param mongo_world: If True the field will work against the mongo world
            instead of the OO world (default: False)
        """
        kwargs = self._extract_marshmallow_field_params(mongo_world)
        if params:
            kwargs.update(params)
        # Retrieve the marshmallow class we inherit from
        for m_class in type(self).mro():
            if (not issubclass(m_class, BaseField) and
                    issubclass(m_class, ma_fields.Field)):
                m_field = m_class(**kwargs)
                # Add i18n support to the field
                m_field.error_messages = I18nErrorDict(m_field.error_messages)
                return m_field
        # Cannot escape the loop given BaseField itself inherits marshmallow's Field


class BaseValidator(ma_validate.Validator):
    """
    All validators in umongo should inherit from this base validator.
    """

    def __init__(self, *args, **kwargs):
        self._error = None
        super().__init__(*args, **kwargs)

    @property
    def error(self):
        return _(self._error)

    @error.setter
    def error(self, value):
        self._error = value

    def as_marshmallow_validator(self, params=None, mongo_world=False):
        """
        Return a pure-marshmallow version of this validator.

        :param params: Additional parameters passed to the mashmallow validator
            class constructor.
        :param mongo_world: If True the field will work against the mongo world
            instead of the OO world (default: False)
        """
        kwargs = {field: getattr(self, field)
                  for field in ('default', 'attribute', 'load_from',
                                'validate', 'required', 'allow_none', 'load_only',
                                'dump_only', 'missing', 'error_messages')}
        if params:
            kwargs.update(params)
        # Retrieve the marshmallow class we inherit from
        for m_class in type(self).mro():
            if (not issubclass(m_class, BaseField) and
                    issubclass(m_class, ma_fields.Field)):
                return m_class(**kwargs)
        # Cannot escape the loop given BaseField itself inherits marshmallow's Field


class BaseDataObject:
    """
    All data objects in umongo should inherit from this base data object.
    """

    def is_modified(self):
        raise NotImplementedError()

    def clear_modified(self):
        raise NotImplementedError()

    @classmethod
    def build_from_mongo(cls, data):
        doc = cls()
        doc.from_mongo(data)
        return doc

    def from_mongo(self, data):
        return self(data)

    def to_mongo(self, update=False):
        return self

    def dump(self):
        return self
