# Escape Mazes Project

# Overview

This project presents using **dynamic programming**
to solve shortest path problem. In this project, there are two
different sets of problems. The first set is the *Known Maze* problem
and the second one is the *Unknown Maze* problem. Those mazes are provided by
The Minigrid library [1].

# Installation

The environments (mazes) can be installed by using the following command:

`pip install minigrid`

# Mazes

We use 7 different mazes in total to test our path finding algorithm.
Here are some of the mazes that we use:

<picture>
  <img src="https://github.com/neilchen1998/escape-the-maze/blob/main/imgs/doorkey-5x5-normal.png" width="300" height="250">
</picture>

- doorkey-5x5-normal

<picture>
  <img src="https://github.com/neilchen1998/escape-the-maze/blob/main/imgs/doorkey-6x6-direct.png" width="300" height="250">
</picture>

- doorkey-6x6-direct

<picture>
  <img src="https://github.com/neilchen1998/escape-the-maze/blob/main/imgs/doorkey-8x8-shortcut.png" width="300" height="250">
</picture>

- doorkey-8x8-shortcut

# Approaches

In this project, we use dynamic programming to solve the shortest path problems.

# Results

| Maze | Steps | Cost |
| ------------- | ------------- | -------- |
| 5x5-normal  | 12  | 12 |
| 6x6-direct  | 4 |4 |
| 6x6-normal  | 15 |15 |
| 6x6-shortcut  | 8 |8 |
| 8x8-direct  | 4 |4 |
| 8x8-normal  | 28 |28 |
| 8x8-shortcut | 12 | 12 |

# Reference

1. [Minigrid Library](https://minigrid.farama.org/)

2. [UCSD ECE276B Planning & Learning in Robotics](https://natanaso.github.io/ece276b/)
