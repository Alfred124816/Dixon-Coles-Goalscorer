import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg


class MatplotlibCanvas(FigureCanvasQTAgg):
    """
    Matplotlib canvas for embedding plots in PyQt5 GUI.
    """
    def __init__(self, width=10, height=8, dpi=100):
        # Create figure with the right size
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        # Add subplot and store reference to axes
        self.axes = self.fig.add_subplot(111)
        
        # Initialize the canvas with the figure
        super(MatplotlibCanvas, self).__init__(self.fig)
        
        # Set up the layout
        self.fig.tight_layout()
    
    def reset(self):
        """
        Reset the figure by clearing and recreating the axes.
        This helps prevent duplicate axes when redrawing.
        """
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.fig.tight_layout()


def create_heatmap(probs, max_goals=5, home_team="Home Team", away_team="Away Team"):
    """
    Create a heatmap visualization of the score probabilities.
    
    Parameters:
    -----------
    probs : np.array
        2D array of probabilities for each scoreline
    max_goals : int
        Maximum number of goals to display
    home_team, away_team : str
        Team names for labels
        
    Returns:
    --------
    fig : matplotlib Figure
        The created figure object
    """
    # Ensure probs is the right shape
    display_probs = probs[:max_goals+1, :max_goals+1]
    
    # Create labels
    home_labels = [str(i) for i in range(max_goals+1)]
    away_labels = [str(i) for i in range(max_goals+1)]
    
    # Convert probabilities to odds
    odds_display = np.zeros_like(display_probs)
    for i in range(display_probs.shape[0]):
        for j in range(display_probs.shape[1]):
            if display_probs[i, j] > 0:
                odds_display[i, j] = 1.0 / display_probs[i, j]
            else:
                odds_display[i, j] = 999.99
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Create heatmap with odds annotations
    ax = sns.heatmap(
        display_probs * 100,  # Still use probabilities for the heatmap colors
        annot=odds_display,  # But annotate with odds
        fmt='.2f',  # Display with 2 decimal places
        cmap='Blues',
        cbar_kws={'label': 'Probability (%)'},
        linewidths=0.5,
        xticklabels=away_labels,
        yticklabels=home_labels
    )
    
    # Set labels and title
    plt.xlabel(f"{away_team} Goals")
    plt.ylabel(f"{home_team} Goals")
    plt.title(f"Score Odds Distribution: {home_team} vs {away_team}")
    
    # Return figure for embedding
    return plt.gcf()


def create_heatmap_for_canvas(canvas, probs, max_goals=5, home_team="Home Team", away_team="Away Team"):
    """
    Create a heatmap visualization on an existing matplotlib canvas.
    
    Parameters:
    -----------
    canvas : MatplotlibCanvas
        Canvas to draw on
    probs : np.array
        2D array of probabilities for each scoreline
    max_goals : int
        Maximum number of goals to display
    home_team, away_team : str
        Team names for labels
    """
    # Reset the canvas to prevent duplicate axes
    canvas.reset()
    
    # Ensure probs is the right shape
    display_probs = probs[:max_goals+1, :max_goals+1]
    
    # Create labels
    home_labels = [str(i) for i in range(max_goals+1)]
    away_labels = [str(i) for i in range(max_goals+1)]
    
    # Convert probabilities to odds
    odds_display = np.zeros_like(display_probs)
    for i in range(display_probs.shape[0]):
        for j in range(display_probs.shape[1]):
            if display_probs[i, j] > 0:
                odds_display[i, j] = 1.0 / display_probs[i, j]
            else:
                odds_display[i, j] = 999.99
    
    # Create heatmap with odds annotations
    sns.heatmap(
        display_probs * 100,  # Still use probabilities for the heatmap colors
        annot=odds_display,  # But annotate with odds
        fmt='.2f',  # Display with 2 decimal places
        cmap='Blues',
        cbar_kws={'label': 'Probability (%)'},
        linewidths=0.5,
        xticklabels=away_labels,
        yticklabels=home_labels,
        ax=canvas.axes
    )
    
    # Set labels and title
    canvas.axes.set_xlabel(f"{away_team} Goals")
    canvas.axes.set_ylabel(f"{home_team} Goals")
    canvas.axes.set_title(f"Score Odds Distribution: {home_team} vs {away_team}")
    
    # Apply tight layout to adjust spacing
    canvas.fig.tight_layout()
    
    # Update canvas
    canvas.draw()


def format_outcome_probabilities(outcomes):
    """
    Format match outcome probabilities for display.
    
    Parameters:
    -----------
    outcomes : dict
        Dictionary with home_win, draw, and away_win probabilities
        
    Returns:
    --------
    str : Formatted string with decimal odds
    """
    home_win_odds = 1 / outcomes['home_win'] if outcomes['home_win'] > 0 else 999.99
    draw_odds = 1 / outcomes['draw'] if outcomes['draw'] > 0 else 999.99
    away_win_odds = 1 / outcomes['away_win'] if outcomes['away_win'] > 0 else 999.99
    
    return f"Home win: {home_win_odds:.2f} | Draw: {draw_odds:.2f} | Away win: {away_win_odds:.2f}"


def format_over_under(over_under, threshold=2.5):
    """
    Format over/under probabilities for display.
    
    Parameters:
    -----------
    over_under : dict
        Dictionary with over and under probabilities
    threshold : float
        Goal threshold
        
    Returns:
    --------
    str : Formatted string with decimal odds
    """
    over_odds = 1 / over_under['over'] if over_under['over'] > 0 else 999.99
    under_odds = 1 / over_under['under'] if over_under['under'] > 0 else 999.99
    
    return f"Over {threshold}: {over_odds:.2f} | Under {threshold}: {under_odds:.2f}"


def generate_sample_data(num_teams=20, num_matches=380):
    """
    Generate sample match data for testing.
    
    Parameters:
    -----------
    num_teams : int
        Number of teams in the dataset
    num_matches : int
        Number of matches to generate
        
    Returns:
    --------
    list : List of (home_team, away_team, home_goals, away_goals) tuples
    """
    teams = [f"Team {i+1}" for i in range(num_teams)]
    
    # Assign random team strengths
    attack_strength = dict(zip(teams, np.random.normal(0, 0.3, num_teams)))
    defense_strength = dict(zip(teams, np.random.normal(0, 0.3, num_teams)))
    
    # Normalize strengths
    attack_mean = np.mean(list(attack_strength.values()))
    defense_mean = np.mean(list(defense_strength.values()))
    
    for team in teams:
        attack_strength[team] -= attack_mean
        defense_strength[team] -= defense_mean
    
    # Generate matches
    matches = []
    
    for _ in range(num_matches):
        home_team, away_team = np.random.choice(teams, 2, replace=False)
        
        # Calculate expected goals
        home_exp = np.exp(0.3 + attack_strength[home_team] - defense_strength[away_team])
        away_exp = np.exp(attack_strength[away_team] - defense_strength[home_team])
        
        # Generate goals
        home_goals = np.random.poisson(home_exp)
        away_goals = np.random.poisson(away_exp)
        
        matches.append((home_team, away_team, home_goals, away_goals))
    
    return matches


def create_player_scoring_heatmap_for_canvas(canvas, player_grid, max_goals=5, home_team="Home Team", away_team="Away Team", player_name="Player"):
    """
    Create a heatmap visualization of player scoring probabilities on an existing matplotlib canvas.
    
    Parameters:
    -----------
    canvas : MatplotlibCanvas
        Canvas to draw on
    player_grid : np.array
        2D array of player scoring probabilities for each scoreline
    max_goals : int
        Maximum number of goals to display
    home_team, away_team : str
        Team names for labels
    player_name : str
        Name of the player for the title
    """
    # Reset the canvas to prevent duplicate axes
    canvas.reset()
    
    # Ensure player_grid is the right shape
    display_grid = player_grid[:max_goals+1, :max_goals+1]
    
    # Create labels
    home_labels = [str(i) for i in range(max_goals+1)]
    away_labels = [str(i) for i in range(max_goals+1)]
    
    # Create heatmap with probability annotations
    sns.heatmap(
        display_grid * 100,  # Convert to percentage for display
        annot=True,
        fmt='.1f',  # Display with 1 decimal place
        cmap='YlOrRd',  # Use YlOrRd colormap to differentiate from match probability heatmap
        cbar_kws={'label': 'Scoring Probability (%)'},
        linewidths=0.5,
        xticklabels=away_labels,
        yticklabels=home_labels,
        ax=canvas.axes
    )
    
    # Set labels and title
    canvas.axes.set_xlabel(f"{away_team} Goals")
    canvas.axes.set_ylabel(f"{home_team} Goals")
    canvas.axes.set_title(f"{player_name} Scoring Probability by Scoreline")
    
    # Apply tight layout to adjust spacing
    canvas.fig.tight_layout()
    
    # Update canvas
    canvas.draw() 