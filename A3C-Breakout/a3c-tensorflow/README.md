# a3c-tensorflow

This repo contains python code for replicating the asynchronous advantage actor-critic algorithm as described in https://arxiv.org/pdf/1602.01783.pdf

## Requirements

* tensorflow
* scipy
* gym (Atari)
* skimage

## Training

For training a3c algorithm in BreakoutDeterministic-v3 using 8 parallel actor learner threads execute the following command:
```
python a3c.py --game BreakoutDeterministic-v3 --num_concurrent 8
```

## Testing

For testing a trained a3c agent execute the folowing command
```
python a3c.py  --game BreakoutDeterministic-v3 --checkpoint_path path_to_checkpoint --testing True
```
## Results

Below you can find 2 plots of training a3c in Breakout and Pong

<img src="https://user-images.githubusercontent.com/9269275/32545061-2b299a12-c483-11e7-8d7b-b1a6c383c34b.png" width="1248">

<img src="https://user-images.githubusercontent.com/9269275/32545062-2b5f4d10-c483-11e7-9d83-05ee962f44e6.png" width="1248">

## Code and Algorithm explanation

Full explanation can be found here: https://papoudakis.github.io/announcements/policy_gradient_a3c/

## Resources

https://github.com/miyosuda/async_deep_reinforce

https://github.com/coreylynch/async-rl

https://github.com/muupan/async-rl
