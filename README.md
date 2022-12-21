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

```jsonl
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
in the `conf` folder. These files show the available options.

Additionally, you can use `hydra` commandline arguments to override specific
values from the configuration files.

TODO: add some examples
