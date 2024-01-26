# inception-bot
Inception-bot is a chess engine that uses neural networks for move generation and position evaluation. It is integrated with lichess-bot, so you can play it on lichess.org.

# Neural Network
The neural network for inception-bot is based on the inception architecture (https://arxiv.org/abs/1409.4842) with a few tweaks including squeeze-exitation layers (https://arxiv.org/abs/1709.01507) and residual connections. It has around ~12 million parameters, consisting of 5 inception blocks.