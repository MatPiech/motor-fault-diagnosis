import cbor2
import click
import numpy as np


@click.command()
@click.option('-f', '--filepath', type=click.Path(), help='Path to data file')
def extract(filepath: str):
    with open(filepath, 'rb') as f:
        data = cbor2.decoder.load(f)

    data = np.array(data['payload']['values'])
    print(data.shape)


if __name__ == '__main__':
    extract()
