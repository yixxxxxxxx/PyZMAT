# src/pyzmat/__init__.py

###################################################################################
#                                                                                 #
#     {_______           {_______ {__{__       {__      {_       {___ {______     #
#     {__    {__                {__  {_ {__   {___     {_ __          {__         #
#     {__    {__{__   {__      {__   {__ {__ { {__    {_  {__         {__         #
#     {_______   {__ {__     {__     {__  {__  {__   {__   {__        {__         #
#     {__          {___     {__      {__   {_  {__  {______ {__       {__         #
#     {__           {__   {__        {__       {__ {__       {__      {__         #
#     {__          {__   {___________{__       {__{__         {__     {__         #
#                {__                                                              #                                                        #
#                                                                                 #
# v0.1.0                                                                          #
# Authors: Yixuan Huang, Benjamin I. Tan                                          #
# Wrapper around ASE and ML-FFs for internal coordinates-based workflows.         #
###################################################################################


from importlib import import_module

__version__ = "0.1.0"

__all__ = [
    "Constraints",
    "ParseUtils",
    "ZmatUtils",
    "PrintUtils",
    "ZMatrix",
]

_LAZY_IMPORTS = {
    "Constraints": ("constraints", "Constraints"),
    "ParseUtils": ("parse_utils", "ParseUtils"),
    "ZmatUtils": ("zmat_utils", "ZmatUtils"),
    "PrintUtils": ("print_utils", "PrintUtils"),
    "ZMatrix": ("zmatrix", "ZMatrix"),
}


def __getattr__(name):
    """Lazily import public objects on first access."""
    try:
        module_name, attr_name = _LAZY_IMPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(list(globals().keys()) + list(__all__))
