# clustering/handler_factory.py

def get_handler(name: str):
    """
    Returns the correct DataHandler subclass based on a string name.
    Example: 'function_1' → Function1DataHandler
    """

    name = name.lower().strip()

    if name == "function_1":
        from function_1.Function1DataHandler import Function1DataHandler
        return Function1DataHandler()

    if name == "function_2":
        from function_2.Function2DataHandler import Function2DataHandler
        return Function2DataHandler()

    if name == "function_3":
        from function_3.Function3DataHandler import Function3DataHandler
        return Function3DataHandler()

    if name == "function_4":
        from function_4.Function4DataHandler import Function4DataHandler
        return Function4DataHandler()

    if name == "function_5":
        from function_5.Function5DataHandler import Function5DataHandler
        return Function5DataHandler()

    if name == "function_6":
        from function_6.Function6DataHandler import Function6DataHandler
        return Function6DataHandler()

    if name == "function_7":
        from function_7.Function7DataHandler import Function7DataHandler
        return Function7DataHandler()

    if name == "function_8":
        from function_8.Function8DataHandler import Function8DataHandler
        return Function8DataHandler()

    raise ValueError(f"Unknown function name: {name}")