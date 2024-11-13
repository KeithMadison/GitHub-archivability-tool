import dash
from dash import dcc, html, Input, Output, State
import plotly.express as px
import requests
import pandas as pd
import dash_bootstrap_components as dbc
from typing import List

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

class GitHubRepoArchiver:
    """Class to handle GitHub API interactions, data processing, and scoring of repositories."""
    
    @staticmethod
    def fetch_repos(org: str) -> pd.DataFrame:
        """Fetch repositories for a given GitHub organization and process the data."""
        url = f'https://api.github.com/orgs/{org}/repos'
        response = requests.get(url)
        
        if response.status_code == 200:
            data = response.json()
            df = pd.DataFrame(data)
            df['updated_at'] = pd.to_datetime(df['updated_at']).dt.tz_localize(None)
            df['archivability_score'] = GitHubRepoArchiver.calculate_archivability_score(df)
            df = df.sort_values(by='archivability_score', ascending=False)
            return df[['name', 'stargazers_count', 'forks_count', 'open_issues_count', 'updated_at', 'archivability_score']]
        else:
            return pd.DataFrame()

    @staticmethod
    def calculate_archivability_score(df: pd.DataFrame) -> pd.Series:
        """Calculate the archivability score for repositories."""
        return (df['stargazers_count'] * 0.4 + 
                df['forks_count'] * 0.3 - 
                df['open_issues_count'] * 0.2 + 
                (pd.Timestamp.now() - df['updated_at']).dt.days * 0.1)

class PlotGenerator:
    """Class to generate different types of plots based on the repository data."""
    
    @staticmethod
    def get_plot(plot_type: str, df: pd.DataFrame):
        """Return the plot corresponding to the plot_type."""
        plot_types = {
            'Stars vs Forks': PlotGenerator.stars_vs_forks,
            'Stars Over Time': PlotGenerator.stars_over_time,
            'Forks vs Issues': PlotGenerator.forks_vs_issues,
            'Histogram of Stars': PlotGenerator.histogram_of_stars,
            'Forks Bar Chart': PlotGenerator.forks_bar_chart,
            'Issues vs Stars Scatter': PlotGenerator.issues_vs_stars
        }
        
        if plot_type in plot_types:
            return plot_types[plot_type](df)
        return None

    @staticmethod
    def stars_vs_forks(df: pd.DataFrame) -> dcc.Graph:
        """Generate Stars vs Forks scatter plot."""
        fig = px.scatter(df, x='stargazers_count', y='forks_count', title='Stars vs Forks')
        return PlotGenerator.style_plot(fig)

    @staticmethod
    def stars_over_time(df: pd.DataFrame) -> dcc.Graph:
        """Generate Stars Over Time line plot."""
        fig = px.line(df.sort_values('updated_at'), x='updated_at', y='stargazers_count', title='Stars Over Time')
        return PlotGenerator.style_plot(fig)

    @staticmethod
    def forks_vs_issues(df: pd.DataFrame) -> dcc.Graph:
        """Generate Forks vs Issues scatter plot."""
        fig = px.scatter(df, x='forks_count', y='open_issues_count', title='Forks vs Issues')
        return PlotGenerator.style_plot(fig)

    @staticmethod
    def histogram_of_stars(df: pd.DataFrame) -> dcc.Graph:
        """Generate histogram of stars count."""
        fig = px.histogram(df, x='stargazers_count', nbins=20, title='Histogram of Stars')
        return PlotGenerator.style_plot(fig)

    @staticmethod
    def forks_bar_chart(df: pd.DataFrame) -> dcc.Graph:
        """Generate Forks Bar Chart."""
        fig = px.bar(df, x='name', y='forks_count', title='Forks Count by Repository')
        return PlotGenerator.style_plot(fig)

    @staticmethod
    def issues_vs_stars(df: pd.DataFrame) -> dcc.Graph:
        """Generate Issues vs Stars scatter plot."""
        fig = px.scatter(df, x='stargazers_count', y='open_issues_count', title='Issues vs Stars')
        return PlotGenerator.style_plot(fig)

    @staticmethod
    def style_plot(fig: px.scatter) -> dcc.Graph:
        """Apply consistent styling to all plots."""
        fig.update_traces(marker=dict(
            opacity=0.8, 
            line=dict(width=1, color='DarkSlateGrey')
        ))

        fig.update_layout(
            title=dict(font=dict(size=18, family='Arial', color='navy'), pad=dict(t=10)),
            margin=dict(l=40, r=40, t=40, b=40),
            plot_bgcolor='whitesmoke',
            paper_bgcolor='white'
        )
        return dcc.Graph(figure=fig)

class RepoRankingTable:
    """Class to generate the table displaying repository archivability ranking."""
    
    @staticmethod
    def generate_table(df: pd.DataFrame) -> dbc.Table:
        """Generate a table to display repository ranking."""
        return dbc.Table(
            [html.Thead(html.Tr([html.Th(col) for col in ['Repository', 'Stars', 'Forks', 'Open Issues', 'Last Updated', 'Archivability Score']])),
             html.Tbody([
                 html.Tr([
                     html.Td(html.A(row['name'], href=f"#{row['name']}", id=f"repo-link-{row['name']}")),
                     html.Td(row['stargazers_count']),
                     html.Td(row['forks_count']),
                     html.Td(row['open_issues_count']),
                     html.Td(row['updated_at']),
                     html.Td(
                         f"{row['archivability_score']:.2f}",
                         style={'background-color': RepoRankingTable.get_archivability_color(row['archivability_score'], df['archivability_score'].min(), df['archivability_score'].max())}
                     )
                 ]) 
                 for _, row in df.iterrows()]
            )],
            bordered=True, hover=True
        )
    
    @staticmethod
    def get_archivability_color(score, min_score, max_score):
        """Map archivability score to color for visual impact."""
        normalized_score = (score - min_score) / (max_score - min_score)
        return f'rgba({int(255 * (1 - normalized_score))}, {int(255 * normalized_score)}, 100, 0.7)'

# App layout
app.layout = dbc.Container([
    html.H1("GitHub Repository Archiving Tool", className="text-center my-4"),

    dbc.Row([
        dbc.Col([
            html.Label("GitHub Organization", className="font-weight-bold"),
            dbc.Input(id='org-input', type='text', placeholder="Enter GitHub organization..."),
            dbc.Button("Fetch Repositories", id='fetch-button', color="primary", className="mt-2"),
            html.Div(id='repo-list', className="mt-3")
        ], width=4, className="d-flex flex-column"),

        dbc.Col([
            html.H3("Plots", className="font-weight-bold"),
            dbc.Row([dbc.Col([html.Div([
                html.H5(f"Plot Section {i+1}", className="font-weight-bold"),
                dcc.Dropdown(
                    id=f"plot-dropdown-{i+1}",
                    options=[{'label': plot, 'value': plot} for plot in ['Stars vs Forks', 'Stars Over Time', 'Forks vs Issues', 'Histogram of Stars', 'Forks Bar Chart', 'Issues vs Stars Scatter']],
                    placeholder=f"Select plot {i+1}",
                    clearable=False
                ),
                html.Div(id=f"plot-container-{i+1}", className="mt-3")
            ], className="border p-2 rounded")], width=4) for i in range(3)], className="d-flex flex-row"),

            html.H3("Repository Archivability Ranking", className="mt-4"),
            html.Div(id='repo-ranking', className="mt-3")
        ], width=8)
    ], className="mb-5"),

], fluid=True)

# Callbacks for interactivity
@app.callback(
    Output('repo-list', 'children'),
    Output('repo-ranking', 'children'),
    Input('fetch-button', 'n_clicks'),
    State('org-input', 'value')
)
def update_repo_list(n_clicks, org_name):
    if n_clicks and org_name:
        df = GitHubRepoArchiver.fetch_repos(org_name)
        if not df.empty:
            top_repos = df.head(5)
            top_repo_names = top_repos['name'].tolist()
            dropdown = dcc.Dropdown(
                id='repo-dropdown',
                options=[{"label": row['name'], "value": row['name']} for _, row in df.iterrows()],
                value=top_repo_names,
                multi=True,
                placeholder="Select repositories",
                searchable=True,
            )
            repo_table = RepoRankingTable.generate_table(df)
            return dropdown, repo_table
        else:
            return html.Div("Organization not found or no repositories available."), html.Div()
    return html.Div(), html.Div()

@app.callback(
    Output('plot-container-1', 'children'),
    Output('plot-container-2', 'children'),
    Output('plot-container-3', 'children'),
    Input('plot-dropdown-1', 'value'),
    Input('plot-dropdown-2', 'value'),
    Input('plot-dropdown-3', 'value'),
    State('org-input', 'value'),
    State('repo-dropdown', 'value')  # Added this input for selected repositories
)
def update_plots(plot1, plot2, plot3, org_name, selected_repos):
    if org_name and selected_repos:  # Check if an organization is entered and repositories are selected
        df = GitHubRepoArchiver.fetch_repos(org_name)
        
        # Filter the dataframe to only include the selected repositories
        df_filtered = df[df['name'].isin(selected_repos)]
        
        plots = []
        for plot in [plot1, plot2, plot3]:
            if plot:
                fig = PlotGenerator.get_plot(plot, df_filtered)  # Use the filtered dataframe
                plots.append(fig)
            else:
                plots.append(html.Div("Please select a plot."))
        
        return plots
    return html.Div(), html.Div(), html.Div()

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
