from torch_geometric.graphgym import register_config


@register_config("extend_dataset_config")
def extend_dataset_config(cfg):
    cfg.dataset.table = None

    cfg.dataset.nodes = "HI-Medium_SSL_Nodes.csv"
    cfg.dataset.edges = "HI-Medium_SSL_Trans.csv"

    # Convert AML amounts to USD with the daily rate of their currency, like the pretraining graphs (etl/aml/fx.py)
    cfg.dataset.convert_currencies = False
    # Daily rates table shared with the ETL, "" for <root_dir>/data/fx_daily_usd.csv (fetched with yfinance if missing)
    cfg.dataset.fx_rates_file = ""

    # Append in/out port numbers (Multi-GNN / MEGA-GNN) as the last two edge features
    cfg.dataset.ports = False

    # Apply sign(x) * log(1 + |x|) to the amount edge features before normalisation
    cfg.dataset.log_amounts = False
