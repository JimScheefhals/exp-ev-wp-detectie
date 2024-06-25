import plotly.express as px
import plotly.graph_objects as go
import numpy as np

from ev_detection.src.features.feature_builder import FeatureBuilder
from ev_detection.src.features.feature_input import FeatureInput
from ev_detection.src.input.load_profiles import LoadProfiles

COLOR_SCALE = px.colors.sequential.Rainbow
def plot_changepoints(_input: FeatureInput, max_profiles: int = 10):

    num_lines = np.min([max_profiles, len(_input.all_profiles)])
    colors = [COLOR_SCALE[i] for i in
              np.linspace(0, len(COLOR_SCALE) - 1, num_lines).astype(int)]

    fig = go.Figure()

    for i in range(num_lines):
        cps = _input.plateaus[i]
        sample = _input.all_profiles[i].reset_index(drop=True)
        color = colors[i]

        fig.add_trace(go.Scatter(
            x=_input.datetime, y=sample,
            mode='lines', line=dict(color=color, width=1.5), name=i, showlegend=True
        ))
        if len(cps) > 0:
            left = [cp[0] for cp in cps]
            right = [cp[1] for cp in cps]
            fig.add_trace(go.Scatter(
                x=_input.datetime.iloc[left], y=sample.iloc[left],
                mode='markers', marker=dict(size=8, color=color, symbol='arrow-bar-right'), showlegend=False
            ))
            fig.add_trace(go.Scatter(
                x=_input.datetime.iloc[right], y=sample.iloc[right],
                mode='markers', marker=dict(size=8, color=color, symbol='arrow-bar-left'), showlegend=False
            ))

    fig.update_layout(
        title="Changepoints",
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
    plot_changepoints(builder.get_input())