# simplerad-backend
simplerad backend fastapi

To run the API:
```bash
./run.sh
```

Access the API at `localhost:8000`. View the documentation on
`localhost:8000/docs`.

### Data
To use the original SimpleRad data, contact @KDercksen.

You can also use your own data. The concepts for entity linking should live in
`data/entity_lists/*.jsonl`, you can use multiple `jsonl` files if you want.
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
in the `conf` folder. These files show the available options.

### OpenAI GPT3 integration (still work in progress...)
To use the `explanation` module (which sits on top of GPT3), you need to put your
OpenAI API key in a `.env` file in the root directory.

```bash
# .env file
OPENAI_API_KEY=sk-...
```
