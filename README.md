# AMKGH (Adaptive Momentum Kurtosis Gated Hybrid) Optimizer

AMKGH introduces a dynamic mechanism that switches between an adaptive update (Adam-style) and a momentum-based update (SGD-style) based on the **kurtosis** (heaviness of tails) of the gradient distribution. This allows it to maintain the fast convergence of adaptive methods on smooth landscapes while providing the robustness of SGD on noisy or ill-conditioned landscapes.

## Prerequisites

To run the code and experiments, you need Python installed along with the following libraries:

- **Python 3.x**
- **PyTorch** (Core deep learning framework)
- **Torchvision** (For MNIST dataset)
- **NumPy** (For math operations)
- **Matplotlib** (For plotting results)
- **Jupyter Notebook** (To execute the `.ipynb` file)

### Installation

You can install the required dependencies using pip:

```
pip install torch torchvision numpy matplotlib notebook
```

## File Structure

- `AMKGH.ipynb`: The main Jupyter Notebook containing the full source code for the optimizer and all benchmark experiments.
    - **Section 1:** Global setup and seeding for reproducibility.
    - **Section 2:** The `AMKGH` Optimizer class implementation.
    - **Section 3-6:** Experiment functions for Quadratic functions, Rosenbrock function, and MNIST.
    - **Section 7:** Main execution block to run all experiments at once.

## Usage

The code is provided as a self-contained Jupyter Notebook.

1. **Open the Notebook:** Launch Jupyter Notebook in your terminal:
    
    ```
    jupyter notebook AMKGH.ipynb
    ```
    
2. **Run All Experiments:** To reproduce all results reported in the paper, you can simply run all cells in the notebook. The final cell contains a `if __name__ == "__main__":` block that automatically triggers:
    - Quadratic Minimization (Condition Number = 10)
    - Quadratic Minimization (Condition Number = 100)
    - Rosenbrock Function Optimization
    - MNIST Neural Network Training
3. **Run Individual Experiments:** You can also run specific experiments interactively by executing the specific cells defining the experiment functions and then calling them manually in a new cell:
    
    ```
    # Run only the Rosenbrock benchmark
    run_rosenbrock()
    ```
    

## Experiments & Benchmarks

The code evaluates AMKGH against three standard baselines: **Adam**, **SGD with Momentum**, and **RMSprop**.

### 1. Quadratic Function

Tests convergence speed on convex surfaces with varying condition numbers ( `κ=10` and `κ=100` ).

- **Goal:** Reach loss `< 10^-6` .

### 2. Rosenbrock Function

Tests ability to escape saddle points and navigate long, narrow, curved valleys (non-convex).

- **Goal:** Reach loss `< 10^-4` .

### 3. MNIST Digit Classification

Tests generalization performance on a real-world deep learning task using a 3-layer MLP (784 -> 128 -> 64 -> 10).

- **Goal:** Maximize test accuracy after 20 epochs.

## Reproducibility

A global seed (`BASE_SEED = 42`) is set at the beginning of the notebook to ensure that weight initialization, data shuffling, and matrix generation are consistent across runs.

## Author

**Lakshya Gupta**

Email: lakshya.gupta.ug24@plaksha.edu.in

---

## License

© 2025 Lakshya Gupta. All rights reserved.

This work is proprietary and confidential. Unauthorized copying, modification, distribution, or use of this software is strictly prohibited without express written permission from the author.