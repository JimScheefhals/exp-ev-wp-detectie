import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from ev_detection.src.features.feature_builder import FeatureBuilder
from ev_detection.src.features.feature_input import FeatureInput
from ev_detection.src.input.load_profiles import LoadProfiles

COLOR_SCALE = px.colors.sequential.Rainbow
def plot_peaks(_input: FeatureInput, max_profiles: int = 10):

    num_lines = np.min([max_profiles, len(_input.all_profiles)])
    colors = [COLOR_SCALE[i] for i in
              np.linspace(0, len(COLOR_SCALE) - 1, num_lines).astype(int)]

    fig = go.Figure()

    for i in range(num_lines):
        properties = _input.peak_properties[i]
        peaks = properties["idx"]
        sample = _input.all_profiles[i].reset_index(drop=True)
        color = colors[i]

        fig.add_trace(go.Scatter(
            x=_input.datetime, y=sample,
            mode='lines', line=dict(color=color, width=1.5), name=i, showlegend=True
        ))
        if len(peaks) > 0:
            fig.add_trace(go.Scatter(
                x=_input.datetime[peaks], y=sample[peaks],
                mode='markers', marker=dict(size=8, color=color), showlegend=False
            ))

    fig.update_layout(
        title="Peaks",
        xaxis_title="Date",
        yaxis_title="Power [kW]",
        showlegend=True
    )

    fig.show()

if __name__ == "__main__":
    syn_profiles = LoadProfiles()
    samples, meta_data = syn_profiles.render_samples(10)
    datetime = LoadProfiles().get_datetimes()
    builder = FeatureBuilder(samples, datetime, meta_data)
    plot_peaks(builder.get_input())