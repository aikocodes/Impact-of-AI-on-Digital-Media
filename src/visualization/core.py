"""
    CS181DV Final Project: Interactive Data Visualization System

    Author: AIKO KATO

    Date: 05/07/2025
    
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output, State
import networkx as nx
import os

# Load preprocessed data
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data_processing', 'cleaned_ai_content.csv')
df = pd.read_csv(DATA_PATH)


# Define color mappings for visuals
# I used ChatGPT for rgb coloring
industry_color_map = {
    'Media': 'rgb(190,186,218)',
    'Marketing': 'rgb(255,255,179)',
    'Education': 'rgb(141,211,199)',
    'Gaming': 'rgb(252,205,229)',
    'Healthcare': 'rgb(253,180,98)',
    'Finance': 'rgb(128,177,211)',
    'Legal': 'rgb(179,222,105)',
    'Automotive': 'rgb(251,128,114)',
    'Retail': 'rgb(217,217,217)',
    'Manufacturing': 'rgb(188,128,189)'
}

regulation_color_map = {
    'Strict': 'rgb(141,211,199)',
    'Moderate': 'rgb(251,128,114)',
    'Lenient': 'rgb(128,177,211)'
}

ai_tool_color_map = {
    'Chatgpt': 'rgb(141,211,199)',
    'Claude': 'rgb(251,128,114)',
    'DALL-E': 'rgb(253,180,98)',
    'Midjourney': 'rgb(128,177,211)',
    'Stable Diffusion': 'rgb(188,128,189)',
    'Synthesia': 'rgb(252,205,229)',
    'Bard': 'rgb(190,186,218)'
}


# Initialize Dash app and server
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server


# Extract unique values for filters
years = sorted(df['Year'].unique())
industries = sorted(df['Industry'].unique())
countries = sorted(df['Country'].unique())
regulations = sorted(df['Regulation Status'].unique())


# App Layout: Tabs with interactive graphs
app.layout = html.Div([
    html.H1("Impact of AI on Digital Media (2020–2025)", style={"textAlign": "center"}),

    dcc.Tabs([
        # 1. Choropleth map
        dcc.Tab(label="AI Metrics by Country", children=[
            dcc.RadioItems(
                id='choropleth-toggle',
                options=[
                    {'label': 'AI Adoption Rate', 'value': 'AI Adoption Rate (%)'},
                    {'label': 'AI Content Volume', 'value': 'AI-Generated Content Volume (TBs per year)'}
                ],
                value='AI Adoption Rate (%)',
                inline=True,
                labelStyle={'margin-right': '15px'}
            ),
            dcc.Slider(
                id='year-slider',
                min=min(years),
                max=max(years),
                value=min(years),
                marks={str(year): str(year) for year in years},
                step=None
            ),
            dcc.Graph(id='choropleth-map')
        ]),
        
        # 2. Animated choropleth map
        dcc.Tab(label="AI Trends Over Time", children=[
            dcc.RadioItems(
                id='animated-metric-toggle',
                options=[
                    {'label': 'AI Adoption Rate', 'value': 'AI Adoption Rate (%)'},
                    {'label': 'AI Content Volume', 'value': 'AI-Generated Content Volume (TBs per year)'}
                ],
                value='AI Adoption Rate (%)',
                inline=True,
                labelStyle={'margin-right': '15px'}
            ),
            dcc.Graph(id='animated-choropleth')
        ]),

        # 3. Line chart
        dcc.Tab(label="AI Adoption Trends by Industry", children=[
            dcc.Dropdown(
                id='industry-dropdown',
                options=[{'label': i, 'value': i} for i in industries],
                value=industries[:3],
                multi=True
            ),
            dcc.Graph(id='industry-line-chart')
        ]),
        
        # 4. Bar chart: Industry content totals
        dcc.Tab(label="AI Content Volume by Industry", children=[
            dcc.Graph(id='industry-content-bar')
        ]),

        # 5. Bar chart: Regulation vs. Trust
        dcc.Tab(label="Regulation vs. Trust", children=[
            dcc.Graph(id='trust-bar-chart')
        ]),
        
        # 6. Scatter plot: Adoption vs. Job Loss or Revenue
        dcc.Tab(label="Adoption vs. Impact", children=[
            dcc.RadioItems(
                id='impact-toggle',
                options=[
                    {'label': 'Job Loss', 'value': 'Job Loss Due to AI (%)'},
                    {'label': 'Revenue Increase', 'value': 'Revenue Increase Due to AI (%)'}
                ],
                value='Job Loss Due to AI (%)',
                inline=True,
                labelStyle={'margin-right': '15px'}
            ),
            dcc.Graph(id='scatter-impact')
        ]),
        
        # 7. Scatter plot + Linked view
        dcc.Tab(label="Linked Industry Explorer", children=[
            dcc.RadioItems(
                id='linked-impact-toggle',
                options=[
                    {'label': 'Job Loss', 'value': 'Job Loss Due to AI (%)'},
                    {'label': 'Revenue Increase', 'value': 'Revenue Increase Due to AI (%)'}
                ],
                value='Job Loss Due to AI (%)',
                inline=True,
                labelStyle={'margin-right': '15px'}
            ),
            dcc.Graph(id='linked-scatter', style={'height': '400px'}),  # I used ChatGPT for this line
            dcc.Graph(id='linked-line')
        ]),
        
        # 8. Bar chart: Tool usage
        dcc.Tab(label="AI Tools Usage", children=[
            dcc.Graph(id='tool-usage-bar')
        ]),
        
        # 9. Network map
        dcc.Tab(label="Tool-Industry Network", children=[
            dcc.Graph(id='tool-network')
        ]),

        # 10. Heatmap
        dcc.Tab(label="Content Volume Heatmap", children=[
            dcc.Graph(id='content-heatmap')
        ])
    ])
])


# Choropleth Callbacks
@app.callback(
    Output('choropleth-map', 'figure'),  # Output: updates the figure in the choropleth-map Graph
    Input('year-slider', 'value'),  # Input: year selected by the slider
    Input('choropleth-toggle', 'value')  # Input: metric selected by radio buttons (adoption or content volume)
)
def update_map(selected_year, selected_metric):
    """
    Generates a choropleth map showing either AI Adoption Rate or AI-Generated Content Volume by country for a selected year

    Args:
        selected_year (int): The year chosen via the slider
        selected_metric (str): The metric selected for visualization (AI Adoption Rate or AI-Generated Content Volume)

    Returns:
        plotly.graph_objs._figure.Figure: Choropleth map figure object
    """
    # Filter dataset for the selected year and group by country to remove duplicates, then compute the mean metric value per country
    filtered = df[df['Year'] == selected_year].groupby("Country", as_index=False)[selected_metric].mean()

    # Create the choropleth figure using Plotly Express
    fig = px.choropleth(
        filtered,  # DataFrame with country-level aggregated values
        locations="Country",
        locationmode="country names",  # Ensure location values match ISO-recognized country names
        color=selected_metric,  # Column to map to color intensity
        color_continuous_scale=[(0, "#ffffff"), (1, "#001c6a")],  # I used ChatGPT for color selection
        title=f"{selected_metric} by Country in {selected_year}"
    )
    return fig


# Animated Choropleth Callbacks
@app.callback(
    Output('animated-choropleth', 'figure'),  # Output: updates the animated map figure
    Input('animated-metric-toggle', 'value')  # Input: metric selection radio buttons
)
def update_animated_choropleth(selected_metric):
    """
    Creates an animated choropleth map showing the selected AI metric across countries over time

    Args:
        selected_metric (str): The metric selected for animation (either AI Adoption Rate or AI-Generated Content Volume)

    Returns:
        plotly.graph_objs._figure.Figure: Animated choropleth figure showing evolution of the selected metric by country
    """
    # Group the dataset by Country and Year, and compute the mean value of the selected metric to avoid duplicates and prepare clean input for animation frames
    df_sorted = df.groupby(['Country', 'Year'], as_index=False)[selected_metric].mean().sort_values('Year')

    # Create the animated choropleth map
    fig = px.choropleth(
        df_sorted,  # Aggregated data
        locations="Country",
        locationmode="country names",
        color=selected_metric,
        animation_frame="Year",  # Enable year-by-year animation
        color_continuous_scale=[[0, "#ffffff"], [1, "#001c6a"]],  # I used ChatGPT for color selection
        range_color=[df[selected_metric].min(), df[selected_metric].max()],  # Fix color scale range across all years
        title=f"{selected_metric} Over Time by Country"
    )
    
    # Adjust plot layout to avoid excessive padding
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    
    return fig


# Line Chart Callbacks
@app.callback(
    Output('industry-line-chart', 'figure'),  # Output: updates the industry line chart figure
    Input('industry-dropdown', 'value')  # Input: list of selected industries from the dropdown
)
def update_line_chart(selected_industries):
    """
    Generates a line chart that visualizes the AI Adoption Rate over time for the industries selected in the dropdown menu

    Args:
        selected_industries (list): A list of industry names selected by the user

    Returns:
        plotly.graph_objs._figure.Figure: Line chart of AI adoption trends by industry
    """
    # Filter the dataset to include only rows where Industry is in the selected list
    filtered = df[df['Industry'].isin(selected_industries)]
    
    # Group by Industry and Year, compute the mean adoption rate for each group
    agg = filtered.groupby(["Industry", "Year"])["AI Adoption Rate (%)"].mean().reset_index()
    
    # Create the line chart using Plotly Express
    fig = px.line(
    agg,  # Aggregated DataFrame
    x="Year",
    y="AI Adoption Rate (%)",
    color="Industry",
    color_discrete_map=industry_color_map,  # Use predefined industry color map
    markers=True,  # Show markers at each data point
    title="AI Adoption Trends by Industry"
    )
    
    return fig


# Bar Chart (Industry content totals) Callbacks
@app.callback(
    Output('industry-content-bar', 'figure'),  # Output: updates the industry content bar chart
    Input('industry-content-bar', 'id')  # Input: ensures the callback runs once at app startup
)
def update_industry_content_bar(_):
    """
    Generates a bar chart showing the total AI-generated content volume for each industry from 2020 to 2025
    The bars are labeled with their percentage share of the total content volume

    Args:
        _ (any): Dummy input (not used) to satisfy Dash callback requirement

    Returns:
        plotly.graph_objs._figure.Figure: Bar chart of content volume by industry
    """
    # Sum total content volume by industry
    content_sum = df.groupby("Industry")["AI-Generated Content Volume (TBs per year)"].sum().reset_index()
    
    # Compute the overall content volume across all industries
    total_volume = content_sum["AI-Generated Content Volume (TBs per year)"].sum()
    
    # Add a new column that represents each industry's share of the total volume
    content_sum["Percentage"] = (content_sum["AI-Generated Content Volume (TBs per year)"] / total_volume) * 100
    
    # Sort industries from highest to lowest total content volume
    content_sum = content_sum.sort_values(by="AI-Generated Content Volume (TBs per year)", ascending=False)

    # Create the bar chart
    fig = px.bar(
        content_sum,  # Aggregated data
        x="Industry",
        y="AI-Generated Content Volume (TBs per year)",
        title="Total AI-Generated Content Volume by Industry (2020–2025)",
        # Label bars with percentage values
        text=content_sum["Percentage"].apply(lambda x: f"{x:.1f}%"),  # I used ChatGPT for this line
        color="Industry",
        color_discrete_map=industry_color_map  # Use predefined industry color palette
    )

    # Customize bar hover info and text label positioning
    fig.update_traces(
        textposition="inside",  # Place percentage labels inside bars
        hovertemplate=
            "Industry: %{x}<br>" +
            "Content Volume: %{y:.2f} TBs<br>" +
            "Share of Total AI Content: %{text}<extra></extra>"
    )

    # Final layout tweaks
    fig.update_layout(showlegend=False, xaxis_tickangle=45)  # Hide legend and rotate x-axis labels for readability
    
    return fig


# Bar Chart (Regulation vs. Consumer Trust) Callbacks
@app.callback(
    Output('trust-bar-chart', 'figure'),  # Output: the bar chart to update
    Input('trust-bar-chart', 'id')  # Input: used to trigger the chart on app load
)
def update_bar(_):
    """
    Generates a bar chart showing the average Consumer Trust in AI for each Regulation Status category across all years

    Args:
        _ (any): Placeholder input (not used in logic); included to trigger the callback when the component is rendered

    Returns:
        plotly.graph_objs._figure.Figure: Bar chart of trust level by regulation status
    """
    # Group the dataset by Regulation Status and compute the average trust level
    grouped = df.groupby("Regulation Status")["Consumer Trust in AI (%)"].mean().reset_index()

    # Create a bar chart with custom colors and percentage labels
    fig = px.bar(
        grouped,  # Aggregated DataFrame
        x="Regulation Status",
        y="Consumer Trust in AI (%)",
        color="Regulation Status",  # Color bars by category
        color_discrete_map=regulation_color_map,  # Use consistent color mapping
        title="Consumer Trust in AI by Regulation Status (2020–2025)",
        # Show trust % on bars
        text=grouped["Consumer Trust in AI (%)"].apply(lambda x: f"{x:.1f}%")  # I used ChatGPT for this line
    )

    # Customize the trace appearance and hover info
    fig.update_traces(
        width=0.4,  # Set bar width
        textposition="inside",  # Place text labels inside bars
        hovertemplate="Regulation Status=%{x}<br>Consumer Trust Rate=%{y:.1f}%<extra></extra>"
    )
    
    # Set Y-axis range and remove legend (not needed here)
    fig.update_layout(
        yaxis=dict(range=[50, 63]),  # Fix y-axis range for consistency
        showlegend=False  # Hide redundant legend
    )
    
    return fig


# Scatter Plot (AI Adoption vs. Impact Metric) Callbacks
@app.callback(
    Output('scatter-impact', 'figure'),  # Output: updates the scatter plot figure
    Input('impact-toggle', 'value')  # Input: selected impact metric (Job Loss or Revenue Increase)
)
def update_scatter(selected_metric):
    """
    Generates a scatter plot comparing AI Adoption Rate to a selected impact metric (either Job Loss Due to AI or Revenue Increase Due to AI), colored by industry

    Args:
        selected_metric (str): The Y-axis metric to visualize; selected by the user from radio buttons

    Returns:
        plotly.graph_objs._figure.Figure: Scatter plot showing the relationship between AI adoption and the selected impact metric
    """
    # Create a scatter plot using the selected Y-axis metric
    fig = px.scatter(
        df,  # Full dataset
        x="AI Adoption Rate (%)",
        y=selected_metric,
        color="Industry",  # Use industry for color grouping
        color_discrete_map=industry_color_map,  # Use defined color palette
        custom_data=["Country", "Year", "Industry"],  # Extra data to show on hover
        title=f"AI Adoption vs. {selected_metric} (2020–2025)"
    )

    # Define a custom hover template using the selected metric name
    hover_template = (
        "Industry: %{customdata[2]}<br>" +
        "Country: %{customdata[0]}<br>" +
        "Year: %{customdata[1]}<br>" +
        "AI Adoption Rate: %{x:.2f}%<br>" +
        f"{selected_metric}: "+"%{y:.2f}%<extra></extra>"
    )

    # Apply the custom hover template to all points
    fig.update_traces(hovertemplate=hover_template)
    
    # Set the figure height for consistent layout
    fig.update_layout(height=500)
    
    return fig


# Linked Scatter Plot Callbacks
@app.callback(
    Output('linked-scatter', 'figure'),  # Output: scatter plot for selection
    Input('linked-impact-toggle', 'value')  # Input: metric to plot on Y-axis (Job Loss or Revenue)
)
def update_linked_scatter(selected_metric):
    """
    Displays a scatter plot of AI Adoption Rate (%) vs. a selected impact metric
    This allows users to select points by industry, and selection will drive the linked line chart

    Args:
        selected_metric (str): Impact metric for Y-axis (Job Loss Due to AI or Revenue Increase Due to AI)

    Returns:
        plotly.graph_objs._figure.Figure: Scatter plot with selectable points
    """
    # Build the scatter plot with industry-colored points
    fig = px.scatter(
        df,  # Full dataset
        x="AI Adoption Rate (%)",
        y=selected_metric,
        color="Industry",
        color_discrete_map=industry_color_map,  # Use industry-specific colors
        hover_data=["Country", "Year"],  # Display extra info on hover
        custom_data=["Industry"],  # Store industry name for selection
        title=f"Select Datapoints for Industry Trend (2020–2025)"
    )
    
    # Enable box or lasso selection mode
    fig.update_layout(dragmode='select')
    
    return fig


# Linked Line Chart Callbacks
@app.callback(
    Output('linked-line', 'figure'),  # Output: linked line chart
    Input('linked-scatter', 'selectedData'),  # Input: selection from scatter plot
    prevent_initial_call=True  # Prevent running before first selection
)
def update_filtered_line(selected_data):
    """
    Generates a line chart showing AI Adoption Rate over time for industries selected in the linked scatter plot

    Args:
        selected_data (dict): The data points selected in the scatter plot

    Returns:
        plotly.graph_objs._figure.Figure: Line chart showing adoption trends for selected industries
    """
    # If nothing is selected, return an empty plot
    if not selected_data or 'points' not in selected_data:
        return go.Figure()

    # Extract unique industries from the selected points
    selected_inds = list({point['customdata'][0] for point in selected_data['points']})
    
    # Filter the main dataset to include only selected industries
    filtered = df[df['Industry'].isin(selected_inds)]
    
    # Group by industry and year to calculate mean adoption rate
    agg = filtered.groupby(["Industry", "Year"])["AI Adoption Rate (%)"].mean().reset_index()

    # Create a line chart to show adoption trends
    fig = px.line(
        agg,  # Aggregated data
        x="Year",
        y="AI Adoption Rate (%)",
        color="Industry",  # Line color by industry
        color_discrete_map=industry_color_map,
        markers=True,
        title="AI Adoption Trends (Filtered by Selection)"
    )
    
    return fig


# Bar Chart (AI Tool Usage Frequency) Callbacks
@app.callback(
    Output('tool-usage-bar', 'figure'),  # Output: updates the bar chart for tool usage
    Input('tool-usage-bar', 'id')  # Input: used to trigger the chart on app load
)
def update_tool_usage_bar(_):
    """
    Generates a bar chart showing how frequently each AI tool is used across industries
    Each bar is labeled with the percentage share of total tool usage

    Args:
        _ (any): Placeholder input (not used); required to trigger the callback on page render

    Returns:
        plotly.graph_objs._figure.Figure: Bar chart showing AI tool usage frequency and percentage
    """
    # Count how many times each AI tool appears in the dataset
    tool_counts = df['Top AI Tools Used'].value_counts().reset_index()
    
    # Rename the columns for clarity
    tool_counts.columns = ['AI Tool', 'Usage Count']
    
    # Calculate total number of tool usages across all rows
    total = tool_counts['Usage Count'].sum()
    
    # Add a column showing each tool's percentage share of total usage
    tool_counts['Percentage'] = (tool_counts['Usage Count'] / total) * 100

    # Create a bar chart showing usage count and percentage per AI tool
    fig = px.bar(
        tool_counts,  # DataFrame of counts and percentages
        x='AI Tool',
        y='Usage Count',
        # Label each bar with percent
        text=tool_counts["Percentage"].apply(lambda x: f"{x:.1f}%"),  # I used ChatGPT for this line
        title="Most Common AI Tools Used Across Industries (2020–2025)",
        color='AI Tool',  # Color bars by tool
        color_discrete_map=ai_tool_color_map  # Use predefined color palette
    )

    # Customize hover and text label behavior for each bar
    fig.update_traces(
        textposition="inside",  # Show percentage inside the bar
        hovertemplate=
            "AI Tool: %{x}<br>" +
            "Usage Count: %{y}<br>" +
            "Share of Tool: %{text}<extra></extra>"  # Custom hover tooltip
    )

    # Final layout tweaks: remove legend and rotate x-axis labels
    fig.update_layout(showlegend=False, xaxis_tickangle=45)  # Hide legend (tool name already shown on x-axis) and rotate labels for readability
    
    return fig


# Network Graph Callbacks(1)
@app.callback(
    Output('tool-network', 'figure'),  # Output: updates the network graph
    Input('tool-network', 'id')  # Input: used to trigger callback on render
)
def update_network(_):
    """
    Constructs and returns a network graph where:
    - Nodes represent either Industries or AI Tools
    - Edges represent usage relationships between industries and tools
    - Edge weight reflects relative frequency of use
    The network is plotted with customized node types, colors, hover info, and interactive legends

    Args:
        _ (any): Placeholder input (not used); required for Dash callback structure

    Returns:
        plotly.graph_objs._figure.Figure: Network graph showing tool-industry relationships
    """
    # Create an undirected graph
    G = nx.Graph()
    
    # Count the number of times each industry uses each AI tool
    edge_weights = df.groupby(['Industry', 'Top AI Tools Used']).size().reset_index(name='count')

    # Normalize edge weights within each industry (so weights are relative)
    industry_totals = edge_weights.groupby('Industry')['count'].transform('sum')
    edge_weights['weight'] = edge_weights['count'] / industry_totals

    # Add nodes and weighted edges to the graph
    for _, row in edge_weights.iterrows():
        G.add_node(row['Industry'], type='Industry')  # Add industry node
        G.add_node(row['Top AI Tools Used'], type='AI Tool')  # Add AI tool node
        G.add_edge(row['Industry'], row['Top AI Tools Used'], weight=row['weight'])  # Connect them

    # Use a spring layout to position nodes in 2D space
    pos = nx.spring_layout(G, seed=42)

    # Create edge traces (lines) representing relationships
    edge_traces = []
    for u, v, attr in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        width = max(1, attr['weight'] * 10)  # Scale edge width by relative weight
        edge_traces.append(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=width, color='lightgray'),  # I used ChatGPT for color selection
            mode='lines',
            hoverinfo='none',
            showlegend=False
        ))

    # Create legend header node placeholders for legend grouping
    node_traces = [
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=14, color='#3EC1D3'),  # I used ChatGPT for color selection
            name='Industry',
            customdata=['header'],
            showlegend=True,
            hoverinfo='skip'
        ),
        go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            marker=dict(size=14, color='#FF6B6B'),  # I used ChatGPT for color selection
            name='AI Tool',
            customdata=['header'],
            showlegend=True,
            hoverinfo='skip'
        ),
        go.Scatter(  # Blank line for spacing in legend
            x=[None],
            y=[None],
            mode='none',
            name=' ',
            showlegend=True,
            hoverinfo='skip'
        )
    ]

    # Sort nodes to ensure stable rendering order
    industry_nodes = sorted([node for node in G.nodes() if G.nodes[node]['type'] == 'Industry'])
    ai_tool_nodes = sorted([node for node in G.nodes() if G.nodes[node]['type'] == 'AI Tool'])

    # Add industry nodes with hover breakdown
    for node in industry_nodes:
        x0, y0 = pos[node]  # Get the (x, y) coordinates of the node from the layout
        
        # Calculate percentage breakdown of tools used by this industry
        breakdown = df[df['Industry'] == node]['Top AI Tools Used'].value_counts(normalize=True) * 100
        
        # Format the hover text with tool percentages
        hovertext = f"{node}:<br>" + "<br>".join([f"{tool}: {pct:.1f}%" for tool, pct in breakdown.items()])
        
        # Create a node trace with text label and hover info for the industry
        node_traces.append(go.Scatter(
            x=[x0],
            y=[y0],
            mode='markers+text',  # Show both marker and text
            text=[node],
            textposition='bottom center',  # Position label under the node
            marker=dict(size=14, color='#3EC1D3'),  # I used ChatGPT for color selection
            hoverinfo='text',  # Enable hover text
            hovertext=[hovertext],
            name=node,
            customdata=['Industry'],  # Used to identify group for toggling
            showlegend=True
        ))

    # Add AI tool nodes with hover breakdown
    for node in ai_tool_nodes:
        x0, y0 = pos[node]  # Get the (x, y) coordinates of the node from the layout
        
        # Calculate percentage breakdown of industries that use this tool
        breakdown = df[df['Top AI Tools Used'] == node]['Industry'].value_counts(normalize=True) * 100
        
        # Format the hover text with industry percentages
        hovertext = f"{node}:<br>" + "<br>".join([f"{ind}: {pct:.1f}%" for ind, pct in breakdown.items()])
        
        # Create a node trace with text label and hover info for the tool
        node_traces.append(go.Scatter(
            x=[x0],
            y=[y0],
            mode='markers+text',
            text=[node],
            textposition='bottom center',
            marker=dict(size=14, color='#FF6B6B'),  # I used ChatGPT for color selection
            hoverinfo='text',
            hovertext=[hovertext],
            name=node,
            customdata=['AI Tool'],
            showlegend=True
        ))

    # Combine edges and nodes into a complete figure
    fig = go.Figure(data=edge_traces + node_traces)
    
    # Configure layout and legend
    fig.update_layout(
        title='AI Tools Used by Industry (2020–2025)',
        margin=dict(l=20, r=20, t=40, b=20),
        height=600,
        legend=dict(
            title="Node Type",  # Legend title
            traceorder='normal'  # Preserve order of legend items
        )
    )
    return fig


# Network Graph Callbacks(2) (Toggle node groups in network via legend click)
@app.callback(
    Output('tool-network', 'figure', allow_duplicate=True),  # Output: updated network figure, allows duplicate callbacks
    Input('tool-network', 'restyleData'),  # Input: legend click data from the user
    State('tool-network', 'figure'),  # State: current figure being shown
    prevent_initial_call=True  # Prevent callback from firing on initial load
)
def toggle_nodes(restyle_data, figure):
    """
    Enables toggling visibility of node groups or individual nodes in the network graph based on user interactions with the legend

    Args:
        restyle_data (dict): Data describing what the user clicked in the legend
        figure (dict): The current state of the tool-network figure

    Returns:
        dict: Updated figure object with adjusted node visibility
    """
    # Return original figure if nothing meaningful was clicked
    if not restyle_data or len(restyle_data) < 2:
        return figure

    # Get the list of trace indices that were clicked
    trace_indices = restyle_data[1]
    if not trace_indices:
        return figure  # Nothing selected

    # Get the actual index of the clicked trace in the figure's data array
    trace_index = trace_indices[0]
    
    # Get the clicked trace object
    clicked_trace = figure['data'][trace_index]
    clicked_name = clicked_trace['name']  # Name shown in the legend
    clicked_customdata = clicked_trace.get('customdata', [None])[0]  # Group type (Industry, AI Tool, or Header)

    # Start with the existing visibility states (defaulting to True if not explicitly set)
    new_visibility = [trace.get('visible', True) for trace in figure['data']]

    # Case 1: Header node was clicked (toggle group)
    if clicked_customdata == 'header':
        group_to_toggle = clicked_name  # This will be either Industry or AI Tool
        
        # Determine what the new visibility should be (True or False)
        header_visible = restyle_data[0].get('visible', [True])[0]
        
        # Toggle all traces belonging to the clicked group
        for i, trace in enumerate(figure['data']):
            trace_customdata = trace.get('customdata', [None])[0]
            if trace_customdata == group_to_toggle:
                new_visibility[i] = header_visible

    # Case 2: Individual node was clicked
    else:
        # Just toggle the visibility of the clicked node
        new_visibility[trace_index] = restyle_data[0].get('visible', [True])[0]

    # Apply updated visibility to the figure
    for i, trace in enumerate(figure['data']):
        trace['visible'] = new_visibility[i]

    return figure


# Heatmap Callbacks
@app.callback(
    Output('content-heatmap', 'figure'),  # Output: updates the heatmap graph
    Input('content-heatmap', 'id')  # Input: used to trigger the callback on app load
)
def update_heatmap(_):
    """
    Generates a heatmap showing AI-generated content volume (in TBs per year) across industries and years
    Each cell reflects the total content volume for a specific industry in a given year

    Args:
        _ (any): Placeholder input (not used); included to trigger the callback when the page loads

    Returns:
        plotly.graph_objs._figure.Figure: Heatmap figure of content volume by industry and year
    """
    # Create a pivot table: Rows = Industry, Columns = Year, Values = Total content volume (sum), Fill missing values with 0
    pivot = df.pivot_table(
        index='Industry',
        columns='Year',
        values='AI-Generated Content Volume (TBs per year)',  # Heat value
        aggfunc='sum'  # Summing volumes across rows
    ).fillna(0)  # Replace NaN with 0 for clarity

    # Generate the heatmap using Plotly Express
    fig = px.imshow(
        pivot,  # Pivot table as input
        labels=dict(color="Content Volume (TBs)"),  # Colorbar label
        aspect="auto",  # Automatically adjust aspect ratio
        color_continuous_scale="YlGnBu",
        title="AI-Generated Content Volume by Industry and Year"
    )
    
    return fig


# Entry Point (Launch dash server)
if __name__ == '__main__':
    # Start the Dash development server with debugging enabled
    app.run_server(debug=True)
