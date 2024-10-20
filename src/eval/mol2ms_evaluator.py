from typing import Tuple

import matplotlib.pyplot as plt
import torch as t


class Mol2MSEvaluator:
    def __init__(self):
        pass

    def evaluate(
        self,
        true_mz: t.Tensor,
        true_intensity: t.Tensor,
        pred_mz: t.Tensor,
        pred_intensity: t.Tensor,
        output_file: str,
    ):
        """
        Evaluate the predicted mass spectrum against the true spectrum and plot the results.
        Negative predictions and ground truths are sorted out.

        Args:
            true_mz (t.Tensor): True mass-to-charge ratios
            true_intensity (t.Tensor): True intensities
            pred_mz (t.Tensor): Predicted mass-to-charge ratios
            pred_intensity (t.Tensor): Predicted intensities
            output_file (str): Path to save the output plot
        """
        # Ensure all inputs are on CPU and converted to numpy for plotting
        true_mz = true_mz.cpu().numpy()
        true_intensity = true_intensity.cpu().numpy()
        pred_mz = pred_mz.cpu().numpy()
        pred_intensity = pred_intensity.cpu().numpy()

        # Filter out negative values
        true_mask = (true_mz >= 0) & (true_intensity >= 0)
        pred_mask = (pred_mz >= 0) & (pred_intensity >= 0)

        true_mz = true_mz[true_mask]
        true_intensity = true_intensity[true_mask]
        pred_mz = pred_mz[pred_mask]
        pred_intensity = pred_intensity[pred_mask]

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot true spectrum
        plt.stem(
            true_mz, true_intensity, linefmt="b-", markerfmt="bo", label="True Spectrum"
        )

        # Plot predicted spectrum
        plt.stem(
            pred_mz,
            pred_intensity,
            linefmt="r-",
            markerfmt="ro",
            label="Predicted Spectrum",
        )

        plt.xlabel("m/z")
        plt.ylabel("Intensity")
        plt.title("True vs Predicted Mass Spectrum")
        plt.legend()

        # Save the plot
        plt.savefig(output_file)
        plt.close()

        print(f"Evaluation plot saved to {output_file}")

    def __call__(
        self,
        true_spectrum: Tuple[t.Tensor, t.Tensor],
        pred_spectrum: Tuple[t.Tensor, t.Tensor],
        output_file: str,
    ):
        """
        Callable interface for the evaluator.

        Args:
            true_spectrum (Tuple[t.Tensor, t.Tensor]): True (mz, intensity) tensors
            pred_spectrum (Tuple[t.Tensor, t.Tensor]): Predicted (mz, intensity) tensors
            output_file (str): Path to save the output plot
        """
        true_mz, true_intensity = true_spectrum
        pred_mz, pred_intensity = pred_spectrum
        self.evaluate(true_mz, true_intensity, pred_mz, pred_intensity, output_file)
