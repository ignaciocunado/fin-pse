import polars as pl
import yfinance as yf


def convert_currencies_to_usd(transactions):
    currencies_to_ticker = {
        'Yen': 'JPYUSD=X',
        'Bitcoin': 'BTC-USD',
        'Mexican Peso': 'MXNUSD=X',
        'UK Pound': 'GBPUSD=X',
        'Rupee': 'INRUSD=X',
        'Yuan': 'CNYUSD=X',
        'Euro': 'EURUSD=X',
        'Australian Dollar': 'AUDUSD=X',
        'Swiss Franc': 'CHFUSD=X',
        'Shekel': 'ILSUSD=X',
        'Canadian Dollar': 'CADUSD=X',
        'Saudi Riyal': 'SARUSD=X',
        'Ruble': 'RUBUSD=X',
        'Brazil Real': 'BRLUSD=X',
    }

    zero = transactions['Timestamp'].min()
    hundred = transactions['Timestamp'].max()

    currencies_to_avg_price = {
      k: yf.Ticker(v).history(start=zero, end=hundred)[['Close']].values.mean() for k,v in currencies_to_ticker.items()
    } | {'US Dollar': 1.0}

    return (
        transactions
        .with_columns(
            (pl.col("Amount Received") * pl.col("Receiving Currency").replace(currencies_to_avg_price).cast(pl.Float64)).alias("Amount")
        ).drop('Amount Paid', 'Amount Received', 'Payment Currency', 'Receiving Currency')
    )


def remove_strings(transactions, currencies=False):
    nodes = (
        transactions
        .select(pl.col('From').alias('Acc'))
        .vstack(transactions.select(pl.col('To').alias('Acc')))
        .unique()
        .with_row_index('Node ID')
    )

    payment_formats = (
        transactions
        .select(pl.col('Payment Format'))
        .unique()
        .with_row_index('format_id')
    )

    final = (
        transactions
        .join(nodes, left_on='From', right_on='Acc')
        .join(nodes, left_on='To', right_on='Acc', suffix='_to')
        .join(payment_formats, on='Payment Format')
        .drop('From', 'To', 'Payment Format',)
        .rename({
            'Node ID': 'From',
            'Node ID_to': 'To',
            'format_id': 'Payment Format',
        })
    )

    if currencies:
        curr = (
            transactions
            .select(pl.col('Receiving Currency').alias('Currency'))
            .vstack(transactions.select(pl.col('Payment Currency').alias('Currency')))
            .unique()
            .with_row_index('currency_id')
        )

        final = (
            final.join(curr, left_on='Payment Currency', right_on='Currency')
            .join(curr, left_on='Receiving Currency', right_on='Currency', suffix='_to')
            .drop('Payment Currency', 'Receiving Currency')
            .rename({
                'currency_id': 'Payment Currency',
                'currency_id_to': 'Receiving Currency',
            })
        )

    return final