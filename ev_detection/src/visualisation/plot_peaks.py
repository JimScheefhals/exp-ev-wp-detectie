import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from ev_detection.src.features.feature_builder import FeatureBuilder
from ev_detection.src.features.feature_input import FeatureInput
from ev_detection.src.input.synthetic_profiles import SyntheticProfiles


def plot_peaks(_input: FeatureInput, max_profiles: int = 10):

    num_lines = np.min([max_profiles, len(_input.all_profiles)])
    colors = [px.colors.sequential.Viridis[i] for i in
              np.linspace(0, len(px.colors.sequential.Viridis) - 1, num_lines).astype(int)]

    fig = go.Figure()

    for i in range(num_lines):
        peaks = _input.peak_properties[i]["idx"]
        sample = _input.all_profiles[i]
        color = colors[i]

        fig.add_trace(go.Scatter(
            x=_input.datetime, y=sample,
            mode='lines', line=dict(color=color, width=1.5)
        ))
        fig.add_trace(go.Scatter(
            x=_input.datetime[peaks], y=sample[peaks],
            mode='markers', marker=dict(size=8, color=color)
        ))

    fig.update_layout(
        title="Peaks",
        xaxis_title="Date",
        yaxis_title="Power [kW]",
        showlegend=False
    )

    fig.show()

if __name__ == "__main__":
    syn_profiles = SyntheticProfiles()
    samples, meta_data = syn_profiles.render_samples(10)
    datetime = SyntheticProfiles().get_datetimes()
    builder = FeatureBuilder(samples, datetime, meta_data)
    plot_peaks(builder.get_input())