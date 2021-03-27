# Brooklyn Bang Bang

> Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

## Approaches

* [Terry Jeffords](agent_code/terry_jeffords/README.md) simple neural network approach
* [Amy Santiago](agent_code/amy_santiago/README.md) is an agent, based on [Pytorch Example](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* [Rosa Diaz](agent_code/rosa_diaz/README.md) agent with pre-selection of valid/invalid and preferred/discouraged actions
* [Jake Peralta](agent_code/jake_peralta/README.md)
* [Nikolaj Boyle](agent_code/nikolaj_boyle/README.md)
* [Scully](agent_code/scully/README.md) agent with full sight but convolution layers in the network
* [Blindfisch](agent_code/blindfisch/README.md) agent with restricted sight and convolution of input features
* [Hitchcock](agent_code/hitchcock/README.md) imitates the rule based agent


## Github Actions

> Training agents can be annoying. To ease our work, we have integrated a Github Action to train our agents automatically. Sadly, my server does not have a graphic card, so we still use the CPU to train. Well, I ain't rich :D 

Run [Github Runner](https://docs.github.com/en/actions/hosting-your-own-runners/about-self-hosted-runners) using a [Docker Runner](https://hub.docker.com/r/tcardonne/github-runner/) image:

```shell
docker run -it --name github-runner \
    -e RUNNER_NAME=private \
    -e GITHUB_ACCESS_TOKEN=... \
    -e RUNNER_TOKEN=... \
    -e RUNNER_REPOSITORY_URL=https://github.com/stefanDeveloper/bomberman \
    -v /var/run/docker.sock:/var/run/docker.sock \
    tcardonne/github-runner:ubuntu-20.04
```

Troubleshooting occurred when we ran the latest image. However, in [this](https://github.com/tcardonne/docker-github-runner/issues/22) pull request, they added the ubuntu support. Using this tag, we can set up our python environment. 

```shell
Error: Version 3.9 with arch x64 not found
```

## Tips & Tricks

Setup environment:

```python
pip install -r requirements.txt
```

Train agent:

```python
python -m main play --agents scully --train 1 --n-round 2500 --no-gui
```

Play agent:

```python
python -m main play --agents scully
```
