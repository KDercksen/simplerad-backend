# simplerad-backend

Installation:

```bash
pip install .
```

To run the API:

```bash
simplerad [configuration]
```

Access the API at `localhost:8000`. View the documentation on
`localhost:8000/docs`.

### Data

To use the original SimpleRad data, contact us personally.

You can also use your own data. The concepts for entity linking should live in
`data/entity_lists/*.jsonl` (can be configured in config files), you can use multiple `jsonl` files if you want.
Each line should be in the following format:

```
{"title": ... , "description": ... , "url": ... , "source": ... , "source_id": ...}
```

You can also add an exclusion list in the same directory called `blacklist`,
with a word/phrase per line. For example:

```txt
klinisch onderzoek
geneeskunding onderzoek
...
```

### Configuration

To configure each available module, check the corresponding configuration file
in the `src/conf` folder. These files show the available options.

Additionally, you can use `hydra` commandline arguments to override specific
values from the configuration files.

#### Examples
The default settings for simplerad are detailed in the `src/conf/config.yaml` file. To override specific defaults, you can use commandline arguments when starting up the API (or alter the configuration files).

```bash
# Overriding the entity recognition module (default simstring)
simplerad entities=flair entities.model_name="/path/to/your/model.pt"
```

Let's say you fork this repository, implement some custom logging prints in the BM25 search module that can be turned on or off. In order to configure this quickly without having to change the config file, use something like this:
```bash
# Overriding the search method to use BM25 (default simstring) and adding some custom argument
simplerad search=bm25 +search.print_logs=True
```
`print_logs` can then be accessed from the `config` object inside the BM25 class.

For more examples on adding/overriding arguments etc, see the `hydra` documentation at https://hydra.cc/docs/intro/
