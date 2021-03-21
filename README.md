# Brooklyn Bang Bang

> Setup for a project/competition amongst students to train a winning Reinforcement Learning agent for the classic game Bomberman.

# Approaches

* [Terry Jeffords](agent_code/terry_jeffords/README.md) simple neural network approach
* [Amy Santiago](agent_code/amy_santiago/README.md) is an agent, based on [Pytorch Example](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
* Rosa Diaz

# Tips & Tricks

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
