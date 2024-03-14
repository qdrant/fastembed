# Contributing to FastEmbed!

:+1::tada: First off, thanks for taking the time to contribute! :tada::+1:

The following is a set of guidelines for contributing to FastEmbed. These are mostly guidelines, not rules. Use your best judgment, and feel free to propose changes to this document in a pull request.

## Table Of Contents

[I don't want to read this whole thing, I just have a question!!!](#i-dont-want-to-read-this-whole-thing-i-just-have-a-question)

[How Can I Contribute?](#how-can-i-contribute)
  * [Your First Code Contribution](#your-first-code-contribution)
  * [Adding New Models](#adding-new-models)

[Styleguides](#styleguides)
  * [Code Lint](#code-lint)
  * [Pre-Commit Hooks](#pre-commit-hooks)
 
## I don't want to read this whole thing I just have a question!!!

> **Note:** Please don't file an issue to ask a question. You'll get faster results by using the resources below:

* [FastEmbed Docs](https://qdrant.github.io/fastembed/)
* [Qdrant Discord](https://discord.gg/Qy6HCJK9Dc)

## How Can I Contribute?

## How Do I Submit A (Good) Bug Report?

Bugs are tracked as [GitHub issues](https://guides.github.com/features/issues/). 

Explain the problem and include additional details to help maintainers reproduce the problem:

* **Use a clear and descriptive title** for the issue to identify the problem.
* **Describe the exact steps which reproduce the problem** in as many details as possible. For example, start by explaining how you are using FastEmbed, e.g. with Langchain, Qdrant Client, Llama Index and which command exactly you used. When listing steps, **don't just say what you did, but explain how you did it**.
* **Provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples. If you're providing snippets in the issue, use [Markdown code blocks](https://help.github.com/articles/markdown-basics/#multiple-lines).
* **Describe the behavior you observed after following the steps** and point out what exactly is the problem with that behavior.
* **Explain which behavior you expected to see instead and why.**
* **If the problem is related to performance or memory**, include a [call stack profile capture](https://github.com/joerick/pyinstrument) and your observations. 

Include details about your configuration and environment:

* **Which version of FastEmbed are you using?** You can get the exact version by running `python -c "import fastembed; print(fastembed.__version__)"`.
* **What's the name and version of the OS you're using**?
* **Which packages do you have installed?** You can get that list by running `pip freeze`

### Your First Code Contribution

Unsure where to begin contributing to FastEmbed? You can start by looking through these `good-first-issue`issues:

* [Good First Issue](https://github.com/qdrant/fastembed/labels/good%20first%20issue) - issues which should only require a few lines of code, and a test or two. These are a great way to get started with FastEmbed. This includes adding new models which are already tested and ready on Huggingface Hub. 

## Pull Requests

The best way to learn about the mechanics of FastEmbed is to start working on it. 

### Your First Code Contribution
Your first code contribution can be small bug fixes:
1. This PR adds a small bug fix for a single input: https://github.com/qdrant/fastembed/pull/148
2. This PR adds a check for the right file location and extension, specific to an OS: https://github.com/qdrant/fastembed/pull/128

Even documentation improvements are most welcome:
1. This PR fixes a README link: https://github.com/qdrant/fastembed/pull/143

### Adding New Models
You can start by adding new models to the FastEmbed. You can find all the model requests [here](https://github.com/qdrant/fastembed/labels/model%20request). 


There are quite a few pull requests that were merged for this purpose and you can use them as a reference. Here is an example: https://github.com/qdrant/fastembed/pull/129 

## Styleguides

### Code Lint
We use ruff for code linting. It should be installed with poetry since it's a dev dependency.

### Pre-Commit Hooks
We use pre-commit hooks to ensure that the code is linted before it's committed. You can install pre-commit hooks by running `pre-commit install` in the root directory of the project.