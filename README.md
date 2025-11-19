# AMKH (Adaptive Momentum Kurtosis Hybrid) Optimizer

This project implements the AMKH optimizer and runs experiments on quadratic functions, Rosenbrock, and MNIST.

Author: Lakshya Gupta, U20240077

## Setup

1. Create a virtual environment:
   ```
   python -m venv venv
   ```

2. Activate the virtual environment:
   ```
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Running the Experiments

Open `AMKH.ipynb` in Jupyter and run the cells. The notebook includes the following experiments:

- **Quadratic Functions**: Tests convergence on well-conditioned (condition number 10) and ill-conditioned (condition number 100) quadratic functions.
- **Rosenbrock Function**: Evaluates performance on the Rosenbrock function, which has a narrow valley that challenges optimizers.
- **MNIST Neural Network**: Trains a simple MLP on the MNIST dataset to compare optimizer performance on a real-world task.

The main execution block runs all experiments sequentially. For the MNIST experiment, the default is 20 epochs; adjust as needed for faster testing.

## Dependencies

- torch
- torchvision
- numpy
- matplotlib

## License

Copyright © 2025 Lakshya Gupta. All Rights Reserved.

This software and associated documentation files (the "Software") are the proprietary and confidential information of Lakshya Gupta. Unauthorized copying, distribution, modification, public display, or public performance of this Software, via any medium, is strictly prohibited.

## Contact

For inquiries, please contact Lakshya Gupta at lakshya.gupta.ug24@plaksha.edu.in