"""Inference Script"""
import os
from torch.utils import data
from src.data_augmentation_time_series.utils.load_synthetic import SyntheticGenerator


def generate_synthetic_data(num_samples=1, tracking_uri='http://0.0.0.0:5005',
                            run_id="e31a5dfda87f4796b3fcc5d401d0a46a"):
    """Main Synthethic Data generation function"""
    syn_generator = SyntheticGenerator(run_id=run_id, tracking_uri=tracking_uri)
    syn_samples = syn_generator.generate_samples(num_samples)
    if len(syn_samples.shape) >= 4:
        # For tts-gan output has 4 dimensions
        syn_samples = syn_samples.squeeze(2)
        syn_samples = syn_samples.transpose(0, 2, 1)

    return syn_samples


if __name__ == "__main__":

    plot_path = "plots"
    os.makedirs(plot_path, exist_ok=True)
    runid = "<ENTER RUN ID HERE>"

    data_syn = generate_synthetic_data(10, run_id=runid)
    if len(data_syn.shape) >= 4:
        data_syn = data_syn.squeeze(2)
        data_syn = data_syn.transpose(0, 2, 1)

    syn_data_loader = data.DataLoader(data_syn, batch_size=1, num_workers=1, shuffle=False)
