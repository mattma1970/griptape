# griptape


## Griptape modifications for running Llama2 local instance rather than via API to LLM providers. 

Griptape's prompt drivers are a great abstraction for call APIs for LLM providers. However, for my projects I want more control over the models, including, using research code bases locally, or simply having the models locally to mitigate the impact of internet outages or poor quality internet in products I want to build. Moreover, I'll be running these models on commodity hardware which are limited in VRAM and so I wanted additional control of the prompt stack to avoid OOM errors which can happen with non-hardened code bases. 

To start, I am targetting Llama 2 7B and added inference clients to Griptape which allow custom endpoints to be passed in and allows direct invocation of the local installation of the model. 
I found that passing a prompt that is too long for the consumer GPU I'm using causes it to hang in a multitude of ways and so in the LocalLlamaPromptDriver I added a 'tail' function that preserves the system prompt, which is at the start of the prompt stack, but passes only the tail of the prompt stack so that the tail contains fewer than the number of tokens that will be problematic. A better approach would be to count tokens on the way into the prompt stack to ammortize the latency of tokenizing on the fly each time.


You'll need to build the custom version of griptape from source. Best to do it in a new virtual environment.
```
python -m venv .venv 
source .venv/bin/activate

git clone https://github.com/mattma1970/griptape.git local_griptape
git checkout local_llama_dev v0.0.1
pip install poetry
cd local_griptape
poetry build
cd dist
pip uninstall griptape
pip install [[insert the name of the .whl file created by the poetry build command]]
```
(also see https://blog.beachgeek.co.uk/getting-started-with-griptape/ for a good griptape hacking post)

You will also need to download the sourcwe code and weight file for llama2. The 7B-chat model has been used to test this code. Go to (llama2)[https://github.com/facebookresearch/llama] and follow the download and ToS acceptance steps. In the current code 

See ```local_griptape/app.py``` for example usage. 

Make sure you update the import statement for Llama in the app.py to point to where to saved it.

----



[![PyPI Version](https://img.shields.io/pypi/v/griptape.svg)](https://pypi.python.org/pypi/griptape)
[![Tests](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/griptape-ai/griptape/actions/workflows/unit-tests.yml)
[![Docs](https://readthedocs.org/projects/griptape/badge/)](https://griptape.readthedocs.io/)
[![Griptape Discord](https://dcbadge.vercel.app/api/server/gnWRz88eym?compact=true&style=flat)](https://discord.gg/gnWRz88eym)

**Griptape** is a modular Python framework for building AI-powered applications that connect securely to your enterprise data and APIs. It offers developers the ability to maintain control and flexibility at every step.

**Build AI Apps**: Easily compose apps in Python with modular structures and ready-made tools. Use built-in drivers to connect to whichever LLMs and data stores you choose.

**Control Data Access**: Connect securely to data sources with granular access controls, ensuring LLMs stay focused on the information that matters.

**Scale With Your Workload**: Easily deploy and run apps in the cloud, where your data lives. Process data ahead of time or vectorize it on the fly.

Using Griptape, you can securely integrate with your internal data stores and APIs. You get to control what data goes into the prompt, and what the LLM is allowed to do with it. 

## Documentation

Please refer to [Griptape Docs](https://docs.griptape.ai/) for:

- Getting started guides. 
- Core concepts and design overviews.
- Examples.
- Contribution guidelines.

Please check out [Griptape Trade School](https://learn.griptape.ai/) for free online courses.

## Quick Start

First, install **griptape**:

```
pip install griptape -U
```

Second, configure an OpenAI client by [getting an API key](https://beta.openai.com/account/api-keys) and adding it to your environment as `OPENAI_API_KEY`. By default, Griptape uses [OpenAI Completions API](https://platform.openai.com/docs/guides/completion) to execute LLM prompts.

With Griptape, you can create *structures*, such as `Agents`, `Pipelines`, and `Workflows`, that are composed of different types of tasks. Let's build a simple creative agent that dynamically uses two tools with shared short-term memory.

```python
from griptape.structures import Agent
from griptape.tools import WebScraper

agent = Agent(
    tools=[WebScraper()]
)

agent.run(
    "based on https://www.griptape.ai/, tell me what Griptape is"
)
```

And here is the output:

> Q: based on https://www.griptape.ai/, tell me what Griptape is  
> A: Griptape is an opinionated Python framework that enables developers to fully harness the potential of LLMs while enforcing strict trust boundaries, schema validation, and activity-level permissions. It offers developers the ability to build AI systems that operate across two dimensions: predictability and creativity. Griptape can be used to create conversational and autonomous agents.

During the run, the Griptape agent loaded a webpage with a **tool**, stored its full content in the **short-term memory**, and finally queried it to answer the original question. The important thing to note here is that no matter how big the webpage is it can never blow up the prompt token limit because the full content never goes back to the main prompt.

[Check out our docs](https://docs.griptape.ai/griptape-framework/structures/prompt-drivers/) to learn more about how to use Griptape with other LLM providers like Anthropic, Claude, Hugging Face, and Azure.

## Versioning

Griptape is in constant development and its APIs and documentation are subject to change. Until we stabilize the API and release version 1.0.0, we will use minor versions (i.e., x.Y.z) to introduce features and breaking features, and patch versions (i.e., x.y.Z) for bug fixes.

## Contributing

Contributions in the form of bug reports, feature ideas, or pull requests are super welcome! Take a look at the current issues and if you'd like to help please submit a pull request with some tests.

## License

Griptape is available under the Apache 2.0 License.
