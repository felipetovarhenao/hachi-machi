<img src="web/static/img/logo.svg" height="160px" />

# hachi machi

To learn more, please visit: https://felipe-tovar-henao.com/hachi-machi

## Development setup

Follow these steps if you intend to run and/or build **hachi machi** from source.

1. Install [Miniconda](https://www.anaconda.com/download/success?reg=skipped).
2. In the terminal, run:

```sh
conda env create -f environment.yml
conda activate hxmx
python -m pip install hatch
```

## On each session

Run to activate:

```sh
conda activate hxmx
```

or to deactivate:

```sh
conda deactivate
```

## Uninstall

```sh
conda remove --name hxmx --all
```

## Build

```sh
hatch build
```

## Publish

```sh
hatch publish
```

## License

**hachi machi** is distributed under the terms of the GNU General Public License version 3 ([GPL-v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html)).
