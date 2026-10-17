import importlib.util
import os.path as osp
import sys

ETL_DIR = osp.join(osp.dirname(osp.abspath(__file__)), "..", "..", "etl", "aml")


def load_etl_module(name: str):
    """Loads etl/aml/<name>.py by path, so the ETL scripts and the data loaders share one implementation (the ETL
    directory is not a package and its scripts import each other by file name)"""
    module_name = f"fin_pse_etl_{name}"
    if module_name not in sys.modules:
        spec = importlib.util.spec_from_file_location(module_name, osp.join(ETL_DIR, f"{name}.py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return sys.modules[module_name]
