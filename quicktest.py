for lib in ["errant", "nltk", "statsmodels", "openpyxl"]:
    try:
        mod = __import__(lib)
        print(f"{lib}: {getattr(mod, '__version__', 'sem __version__')}")
    except ModuleNotFoundError:
        print(f"{lib}: NÃO INSTALADO")