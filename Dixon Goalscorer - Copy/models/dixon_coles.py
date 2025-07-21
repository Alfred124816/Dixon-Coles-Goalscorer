import numpy as np
from scipy import stats
from scipy.optimize import minimize
import math


class DixonColesModel:
    def __init__(self, max_goals=10):
        """
        Dixon-Coles model for predicting soccer match outcomes.
        
        Parameters:
        -----------
        max_goals : int
            Maximum number of goals to consider in the model (default: 10)
        """
        self.max_goals = max_goals
        self.params = None
        self.team_attack = {}
        self.team_defense = {}
        self.home_advantage = None
        self.rho = None
        
    def _poisson_probability(self, home_mean, away_mean, home_goals, away_goals, rho):
        """
        Calculate the probability of a specific scoreline using the Dixon-Coles model.
        
        Parameters:
        -----------
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        home_goals, away_goals : int
            The number of goals scored
        rho : float
            Correlation parameter for low-scoring adjustment
            
        Returns:
        --------
        float : Probability of specified scoreline
        """
        lambda_home = home_mean
        lambda_away = away_mean
        
        # Apply Dixon-Coles adjustment for specific low-scoring results
        if home_goals == 0 and away_goals == 0:
            dc_adj = 1 - (lambda_home * lambda_away * rho)
        elif home_goals == 0 and away_goals == 1:
            dc_adj = 1 + (lambda_home * rho)
        elif home_goals == 1 and away_goals == 0:
            dc_adj = 1 + (lambda_away * rho)
        elif home_goals == 1 and away_goals == 1:
            dc_adj = 1 - rho
        else:
            dc_adj = 1.0
            
        p = stats.poisson.pmf(home_goals, lambda_home) * stats.poisson.pmf(away_goals, lambda_away) * dc_adj
        
        return p
    
    def fit(self, match_data):
        """
        Fit the Dixon-Coles model to historical match data.
        This is still kept for compatibility but not used in user-input mode.
        
        Parameters:
        -----------
        match_data : list of tuples
            List of (home_team, away_team, home_goals, away_goals) tuples
            
        Returns:
        --------
        self
        """
        # Get unique teams
        teams = set()
        for home, away, _, _ in match_data:
            teams.add(home)
            teams.add(away)
        teams = sorted(list(teams))
        
        # Initial parameter values
        n_teams = len(teams)
        x0 = np.zeros(2 * n_teams + 2)
        x0[0] = 0.1  # home advantage
        x0[1] = -0.16  # rho parameter
        
        # Constraint: sum of attack and defense parameters = 0
        def constraint_attack(x):
            return np.sum(x[2:2+n_teams])
            
        def constraint_defense(x):
            return np.sum(x[2+n_teams:2+2*n_teams])
            
        constraints = [
            {'type': 'eq', 'fun': constraint_attack},
            {'type': 'eq', 'fun': constraint_defense}
        ]
        
        # Negative log-likelihood function
        def neg_log_likelihood(x):
            home_advantage = x[0]
            rho = x[1]
            attack_params = x[2:2+n_teams]
            defense_params = x[2+n_teams:2+2*n_teams]
            
            # Create team parameter dictionaries
            team_attack = dict(zip(teams, attack_params))
            team_defense = dict(zip(teams, defense_params))
            
            log_likelihood = 0
            for home, away, home_goals, away_goals in match_data:
                # Calculate expected goals
                lambda_home = np.exp(home_advantage + team_attack[home] + team_defense[away])
                lambda_away = np.exp(team_attack[away] + team_defense[home])
                
                # Apply Dixon-Coles adjustments for low scores and draw scores
                # Calculate Poisson probabilities
                home_prob = stats.poisson.pmf(home_goals, lambda_home)
                away_prob = stats.poisson.pmf(away_goals, lambda_away)
                
                # Apply low-scoring adjustment (Dixon-Coles)
                if home_goals == 0 and away_goals == 0:
                    adjustment = 1 - (lambda_home * lambda_away * rho)
                elif home_goals == 0 and away_goals == 1:
                    adjustment = 1 + (lambda_home * rho)
                elif home_goals == 1 and away_goals == 0:
                    adjustment = 1 + (lambda_away * rho)
                elif home_goals == 1 and away_goals == 1:
                    adjustment = 1 - rho
                else:
                    adjustment = 1
                
                p = home_prob * away_prob * adjustment
                
                log_likelihood += np.log(p) if p > 0 else -10  # Avoid log(0)
                
            return -log_likelihood
            
        # Optimize parameters
        bounds = [(0, None)] + [(-0.2, 0.2)] + [(-1, 1)] * (2 * n_teams)
        result = minimize(neg_log_likelihood, x0, method='SLSQP', constraints=constraints, bounds=bounds)
        
        # Store optimized parameters
        self.home_advantage = result.x[0]
        self.rho = result.x[1]
        self.team_attack = dict(zip(teams, result.x[2:2+n_teams]))
        self.team_defense = dict(zip(teams, result.x[2+n_teams:2+2*n_teams]))
        
        return self
    
    def predict_score_grid(self, home_mean, away_mean, max_goals=None, rho=-0.16):
        """
        Generate a grid of probabilities for all scorelines up to max_goals.
        
        Parameters:
        -----------
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        max_goals : int, optional
            Maximum number of goals to consider for each team.
            If None, uses self.max_goals.
        rho : float
            Correlation parameter for low-scoring adjustment
            
        Returns:
        --------
        np.array : 2D array of probabilities for each scoreline
        """
        if max_goals is None:
            max_goals = self.max_goals
            
        # Vectorized implementation
        # Create a grid of all possible scorelines
        home_goals, away_goals = np.meshgrid(
            np.arange(max_goals + 1),
            np.arange(max_goals + 1),
            indexing='ij'
        )
        
        # Calculate Poisson probabilities for each team (vectorized)
        home_probs = stats.poisson.pmf(home_goals, home_mean)
        away_probs = stats.poisson.pmf(away_goals, away_mean)
        
        # Apply Dixon-Coles adjustment (vectorized)
        # Initialize adjustment matrix with ones
        adjustment = np.ones((max_goals + 1, max_goals + 1))
        
        # Apply adjustment only to the specific low scores according to Dixon-Coles paper
        # (0,0), (0,1), (1,0), (1,1)
        adjustment[0, 0] = 1 - (home_mean * away_mean * rho)
        adjustment[0, 1] = 1 + (home_mean * rho)
        adjustment[1, 0] = 1 + (away_mean * rho)
        adjustment[1, 1] = 1 - rho
        
        # Calculate final probability grid with adjustment
        grid = home_probs * away_probs * adjustment
        
        # Normalize the grid
        grid = grid / np.sum(grid)
        
        return grid
    
    def predict_match_outcome(self, home_mean, away_mean, rho=-0.16):
        """
        Calculate the probabilities of home win, draw, and away win.
        
        Parameters:
        -----------
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        rho : float
            Correlation parameter for low-scoring adjustment
            
        Returns:
        --------
        dict : Dictionary with probabilities for home win, draw, and away win
        """
        # Get the score grid using the vectorized method
        grid = self.predict_score_grid(home_mean, away_mean, self.max_goals, rho)
                
        # Calculate outcome probabilities using NumPy operations
        # Home win: sum of values where home goals > away goals (lower triangle excluding diagonal)
        home_win = np.sum(np.tril(grid, -1))
        
        # Draw: sum of diagonal values
        draw = np.sum(np.diag(grid))
        
        # Away win: sum of values where away goals > home goals (upper triangle excluding diagonal)
        away_win = np.sum(np.triu(grid, 1))
        
        return {
            'home_win': home_win,
            'draw': draw,
            'away_win': away_win
        }
    
    def predict_double_chance(self, home_mean, away_mean, rho=-0.16):
        """
        Calculate the probabilities of double chance outcomes.
        
        Parameters:
        -----------
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        rho : float
            Correlation parameter for low-scoring adjustment
            
        Returns:
        --------
        dict : Dictionary with probabilities for 1X, X2, 12
        """
        outcomes = self.predict_match_outcome(home_mean, away_mean, rho)
        
        return {
            'home_draw': outcomes['home_win'] + outcomes['draw'],    # 1X
            'draw_away': outcomes['draw'] + outcomes['away_win'],    # X2
            'home_away': outcomes['home_win'] + outcomes['away_win'] # 12
        }
        
    def predict_over_under(self, home_mean, away_mean, threshold=2.5, rho=-0.16):
        """
        Calculate the probability of total goals being over or under a threshold.
        
        Parameters:
        -----------
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        threshold : float
            Goal threshold (default: 2.5)
        rho : float
            Correlation parameter for low-scoring adjustment
            
        Returns:
        --------
        dict : Dictionary with probabilities for over and under
        """
        # Get the score grid using the vectorized method
        grid = self.predict_score_grid(home_mean, away_mean, self.max_goals, rho)
        
        # Create total goals matrix (i+j for each cell)
        home_goals, away_goals = np.meshgrid(
            np.arange(self.max_goals + 1),
            np.arange(self.max_goals + 1),
            indexing='ij'
        )
        total_goals = home_goals + away_goals
        
        # Create under/over masks
        under_mask = total_goals < threshold
        over_mask = ~under_mask  # Equivalent to total_goals >= threshold
        
        # Calculate probabilities using masks
        under_prob = np.sum(grid[under_mask])
        over_prob = np.sum(grid[over_mask])
                    
        return {
            'over': over_prob,
            'under': under_prob
        }

    def predict_team_over_under(self, team_mean, threshold=2.5):
        """
        Calculate the probability of a team scoring over or under a threshold.
        
        Parameters:
        -----------
        team_mean : float
            Expected goals (mean) for the team
        threshold : float
            Goal threshold
            
        Returns:
        --------
        dict : Dictionary with probabilities for over and under
        """
        # Use simple Poisson for individual team scores
        under_prob = stats.poisson.cdf(threshold - 0.5, team_mean)
        over_prob = 1 - under_prob
        
        return {
            'over': over_prob,
            'under': under_prob
        }
    
    def predict_btts(self, home_mean, away_mean, rho=-0.16):
        """
        Calculate the probability of both teams to score (BTTS).
        
        Parameters:
        -----------
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        rho : float
            Correlation parameter for low-scoring adjustment
            
        Returns:
        --------
        dict : Dictionary with probabilities for yes and no
        """
        # Get the score grid using the vectorized method
        grid = self.predict_score_grid(home_mean, away_mean, self.max_goals, rho)
        
        # Create coordinate matrices
        home_goals, away_goals = np.meshgrid(
            np.arange(self.max_goals + 1),
            np.arange(self.max_goals + 1),
            indexing='ij'
        )
        
        # Create mask for BTTS: home_goals > 0 AND away_goals > 0
        btts_mask = (home_goals > 0) & (away_goals > 0)
        
        # Calculate BTTS probability using mask
        btts_yes = np.sum(grid[btts_mask])
        btts_no = 1 - btts_yes
        
        return {
            'yes': btts_yes,
            'no': btts_no
        }
    
    def predict_btts_and_over_under(self, home_mean, away_mean, threshold=2.5, rho=-0.16):
        """
        Calculate the probability of BTTS combined with over/under.
        
        Parameters:
        -----------
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        threshold : float
            Goal threshold (default: 2.5)
        rho : float
            Correlation parameter for low-scoring adjustment
            
        Returns:
        --------
        dict : Dictionary with probabilities for different combinations
        """
        # Get the score grid using the vectorized method
        grid = self.predict_score_grid(home_mean, away_mean, self.max_goals, rho)
        
        # Create coordinate matrices
        home_goals, away_goals = np.meshgrid(
            np.arange(self.max_goals + 1),
            np.arange(self.max_goals + 1),
            indexing='ij'
        )
        
        # Create total goals matrix
        total_goals = home_goals + away_goals
        
        # Create BTTS mask: home_goals > 0 AND away_goals > 0
        btts_mask = (home_goals > 0) & (away_goals > 0)
        no_btts_mask = ~btts_mask
        
        # Create over/under masks
        over_mask = total_goals >= threshold
        under_mask = ~over_mask
        
        # Calculate probabilities for each combination using masks
        yes_over_mask = btts_mask & over_mask
        yes_under_mask = btts_mask & under_mask
        no_over_mask = no_btts_mask & over_mask
        no_under_mask = no_btts_mask & under_mask
        
        # Sum probabilities for each combination
        yes_over = np.sum(grid[yes_over_mask])
        yes_under = np.sum(grid[yes_under_mask])
        no_over = np.sum(grid[no_over_mask])
        no_under = np.sum(grid[no_under_mask])
        
        return {
            'yes_over': yes_over,
            'yes_under': yes_under,
            'no_over': no_over,
            'no_under': no_under
        }
    
    def predict_player_anytime_scorer(self, player_mean, team_mean):
        """
        Calculate the probability of a player scoring at any time during the match.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals (mean) for the player
        team_mean : float
            Expected goals (mean) for the player's team
            
        Returns:
        --------
        float : Probability of the player scoring at any time
        """
        # Probability of scoring at least one goal is 1 - probability of scoring no goals
        if team_mean <= 0:
            return 0
            
        # Calculate player's contribution ratio
        player_ratio = player_mean / team_mean
        
        # Calculate probability that player doesn't score when team scores (for any team goals)
        # Using Poisson distribution for team goals, but binomial for player's share
        prob_not_score = 0
        
        # Calculate for each possible number of team goals (up to max_goals)
        for i in range(self.max_goals + 1):
            # Probability team scores i goals
            prob_team_i_goals = stats.poisson.pmf(i, team_mean)
            
            # Probability player doesn't score any of those i goals
            if i == 0:
                prob_player_not_score = 1  # If team doesn't score, player can't score
            else:
                prob_player_not_score = (1 - player_ratio) ** i
                
            # Add to total probability
            prob_not_score += prob_team_i_goals * prob_player_not_score
            
        # Probability of scoring is 1 - probability of not scoring
        return 1 - prob_not_score
        
    def predict_player_scorer_and_result(self, player_mean, team_mean, home_mean, away_mean, 
                                         is_home_team=True, rho=-0.16, score_grid=None):
        """
        Calculate probabilities for player scoring combined with match result.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals (mean) for the player
        team_mean : float
            Expected goals (mean) for the player's team (either home or away)
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        is_home_team : bool
            True if player is on the home team, False if on away team
        rho : float
            Correlation parameter for low-scoring adjustment
        score_grid : np.array, optional
            Pre-calculated scoreline probability grid. If provided, used instead of recalculating.
            
        Returns:
        --------
        dict : Dictionary with probabilities for different combinations
        """
        if team_mean <= 0:
            return {
                'scores_team_wins': 0,
                'scores_team_draws': 0,
                'scores_team_loses': 0,
                'no_score_team_wins': 0,
                'no_score_team_draws': 0,
                'no_score_team_loses': 0
            }
            
        # First calculate the anytime scorer probability as an upper bound
        anytime_scorer_prob = self.predict_player_scorer(player_mean, team_mean, score_grid, is_home_team)
            
        # Player's contribution ratio
        player_ratio = player_mean / team_mean
        
        # Get the score grid
        if score_grid is None:
            # If not provided, calculate the Dixon-Coles probability grid
            grid = np.zeros((self.max_goals + 1, self.max_goals + 1))
            
            for i in range(self.max_goals + 1):
                for j in range(self.max_goals + 1):
                    grid[i, j] = self._poisson_probability(home_mean, away_mean, i, j, rho)
                    
            # Normalize the grid
            grid = grid / np.sum(grid)
                
            # Use self.max_goals for iteration
            max_goals = self.max_goals
        else:
            # Use the provided score grid
            grid = score_grid
            
            # Use the grid's dimensions for iteration
            max_goals = min(grid.shape[0] - 1, grid.shape[1] - 1)
        
        # Initialize result probabilities
        scores_team_wins = 0
        scores_team_draws = 0
        scores_team_loses = 0
        no_score_team_wins = 0
        no_score_team_draws = 0
        no_score_team_loses = 0
        
        # Calculate probabilities
        for i in range(max_goals + 1):  # home goals
            for j in range(max_goals + 1):  # away goals
                # Determine match result
                if i > j:  # Home win
                    if is_home_team:
                        team_goals = i
                        result_type = "win"
                    else:
                        team_goals = j
                        result_type = "loss"
                elif i == j:  # Draw
                    if is_home_team:
                        team_goals = i
                    else:
                        team_goals = j
                    result_type = "draw"
                else:  # Away win
                    if is_home_team:
                        team_goals = i
                        result_type = "loss"
                    else:
                        team_goals = j
                        result_type = "win"
                
                # Probability of the scoreline
                scoreline_prob = grid[i, j]
                
                # Calculate probability that player scores at least once given team scored team_goals
                if team_goals == 0:
                    prob_player_scores = 0  # Player can't score if team doesn't score
                else:
                    # Probability player scores at least one of the team's goals
                    prob_player_scores = 1 - (1 - player_ratio) ** team_goals
                
                # Probability player doesn't score given this scoreline
                prob_player_not_score = 1 - prob_player_scores
                
                # Update result probabilities
                if result_type == "win":
                    scores_team_wins += scoreline_prob * prob_player_scores
                    no_score_team_wins += scoreline_prob * prob_player_not_score
                elif result_type == "draw":
                    scores_team_draws += scoreline_prob * prob_player_scores
                    no_score_team_draws += scoreline_prob * prob_player_not_score
                else:  # loss
                    scores_team_loses += scoreline_prob * prob_player_scores
                    no_score_team_loses += scoreline_prob * prob_player_not_score
        
        # Create the result dictionary
        result = {
            'scores_team_wins': scores_team_wins,
            'scores_team_draws': scores_team_draws,
            'scores_team_loses': scores_team_loses,
            'no_score_team_wins': no_score_team_wins,
            'no_score_team_draws': no_score_team_draws,
            'no_score_team_loses': no_score_team_loses
        }
        
        # Ensure the scoring probabilities don't exceed the anytime scorer probability
        scorer_keys = ['scores_team_wins', 'scores_team_draws', 'scores_team_loses']
        
        # First ensure individual probabilities don't exceed anytime scorer probability
        for key in scorer_keys:
            result[key] = min(result[key], anytime_scorer_prob)
            
        # Then ensure their sum doesn't exceed anytime scorer probability
        total_scorer_prob = sum(result[key] for key in scorer_keys)
        if total_scorer_prob > anytime_scorer_prob:
            ratio = anytime_scorer_prob / total_scorer_prob
            for key in scorer_keys:
                result[key] *= ratio
        
        return result
    
    def predict_player_scorer_and_double_chance(self, player_mean, team_mean, home_mean, away_mean, 
                                               is_home_team=True, rho=-0.16):
        """
        Calculate probabilities for player scoring combined with double chance results.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals (mean) for the player
        team_mean : float
            Expected goals (mean) for the player's team (either home or away)
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        is_home_team : bool
            True if player is on the home team, False if on away team
        rho : float
            Correlation parameter for low-scoring adjustment
            
        Returns:
        --------
        dict : Dictionary with probabilities for different combinations
        """
        # Get single result probabilities
        single_results = self.predict_player_scorer_and_result(
            player_mean, team_mean, home_mean, away_mean, is_home_team, rho
        )
        
        # Calculate double chance probabilities
        if is_home_team:
            dc_results = {
                'scores_1X': single_results['scores_team_wins'] + single_results['scores_team_draws'],
                'scores_X2': single_results['scores_team_draws'] + single_results['scores_team_loses'],
                'scores_12': single_results['scores_team_wins'] + single_results['scores_team_loses'],
                'no_score_1X': single_results['no_score_team_wins'] + single_results['no_score_team_draws'],
                'no_score_X2': single_results['no_score_team_draws'] + single_results['no_score_team_loses'],
                'no_score_12': single_results['no_score_team_wins'] + single_results['no_score_team_loses']
            }
        else:
            dc_results = {
                'scores_1X': single_results['scores_team_loses'] + single_results['scores_team_draws'],
                'scores_X2': single_results['scores_team_draws'] + single_results['scores_team_wins'],
                'scores_12': single_results['scores_team_loses'] + single_results['scores_team_wins'],
                'no_score_1X': single_results['no_score_team_loses'] + single_results['no_score_team_draws'],
                'no_score_X2': single_results['no_score_team_draws'] + single_results['no_score_team_wins'],
                'no_score_12': single_results['no_score_team_loses'] + single_results['no_score_team_wins']
            }
    
        # First calculate the anytime scorer probability as an upper bound
        anytime_scorer_prob = self.predict_player_scorer(player_mean, team_mean, None, is_home_team)
        
        # Ensure the scoring probabilities don't exceed the anytime scorer probability
        scorer_keys = ['scores_1X', 'scores_X2', 'scores_12']
        
        # First ensure individual probabilities don't exceed anytime scorer probability
        for key in scorer_keys:
            dc_results[key] = min(dc_results[key], anytime_scorer_prob)
            
        # No need to adjust the sum since these are already combinations of the adjusted
        # values from predict_player_scorer_and_result
        
        return dc_results
    
    def predict_player_exact_goals(self, player_mean, team_mean, exact_goals, score_grid=None, is_home_team=True):
        """
        Calculate the probability of a player scoring exactly a specific number of goals.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals (mean) for the player
        team_mean : float
            Expected goals (mean) for the player's team
        exact_goals : int
            The exact number of goals to calculate probability for
        score_grid : np.array, optional
            Pre-calculated scoreline probability grid. If provided, used instead of recalculating.
        is_home_team : bool
            Whether the player is on the home team (True) or away team (False)
            
        Returns:
        --------
        float : Probability of the player scoring exactly this many goals
        """
        if exact_goals < 0:
            return 0
            
        if team_mean <= 0:
            return 1.0 if exact_goals == 0 else 0.0
            
        # If exact_goals is 0, it's a special case
        if exact_goals == 0:
            # If score grid is provided, use vectorized calculation
            if score_grid is not None:
                # Get the shape of the grid
                max_goals = score_grid.shape[0] - 1
                
                # Create team goals matrix based on player's team
                if is_home_team:
                    team_goals_matrix = np.arange(max_goals + 1)[:, np.newaxis]  # Column vector
                    grid_to_use = score_grid
                else:
                    team_goals_matrix = np.arange(max_goals + 1)[np.newaxis, :]  # Row vector
                    grid_to_use = score_grid.T
                
                # Calculate probability that player doesn't score for each team goal count
                player_ratio = player_mean / team_mean
                player_no_score_probs = np.power(1 - player_ratio, team_goals_matrix)
                
                # Multiply by scoreline probabilities and sum
                if is_home_team:
                    total_prob = np.sum(grid_to_use * player_no_score_probs)
                else:
                    total_prob = np.sum(grid_to_use * player_no_score_probs)
                    
                return total_prob
            else:
                # Calculate probability that player doesn't score when team scores (for any team goals)
                prob_not_score = 0
                
                # Calculate for each possible number of team goals (up to max_goals)
                for i in range(self.max_goals + 1):
                    # Probability team scores i goals
                    prob_team_i_goals = stats.poisson.pmf(i, team_mean)
                    
                    # Probability player doesn't score any of those i goals
                    if i == 0:
                        prob_player_not_score = 1  # If team doesn't score, player can't score
                    else:
                        prob_player_not_score = (1 - player_mean/team_mean) ** i
                        
                    # Add to total probability
                    prob_not_score += prob_team_i_goals * prob_player_not_score
                    
                return prob_not_score
        
        # For non-zero exact_goals
        # Player's contribution ratio
        player_ratio = player_mean / team_mean
        
        # If score grid is provided, use vectorized calculation
        if score_grid is not None:
            # Get the shape of the grid
            max_goals = score_grid.shape[0] - 1
            
            # Create team goals matrix based on player's team
            if is_home_team:
                team_goals_matrix = np.arange(max_goals + 1)[:, np.newaxis]  # Column vector
                grid_to_use = score_grid
            else:
                team_goals_matrix = np.arange(max_goals + 1)[np.newaxis, :]  # Row vector
                grid_to_use = score_grid.T
            
            # Only consider team goals ≥ exact_goals
            # (player can't score more goals than their team)
            valid_mask = (team_goals_matrix >= exact_goals)
            
            if not np.any(valid_mask):
                return 0.0
            
            # Create matrix for binomial probabilities
            player_exact_probs = np.zeros_like(team_goals_matrix, dtype=float)
            
            # Calculate binomial probability for each valid team goal count
            team_goals_values = team_goals_matrix[valid_mask]
            player_exact_probs[valid_mask] = stats.binom.pmf(
                exact_goals,
                team_goals_values,
                player_ratio
            )
            
            # Multiply by scoreline probabilities and sum
            if is_home_team:
                total_prob = np.sum(grid_to_use * player_exact_probs)
            else:
                total_prob = np.sum(grid_to_use * player_exact_probs)
                
            return total_prob
        else:
            # Calculate probability for each possible number of team goals
            total_prob = 0
            
            for i in range(exact_goals, self.max_goals + 1):
                # Probability team scores i goals
                prob_team_i_goals = stats.poisson.pmf(i, team_mean)
                
                # Probability player scores exactly exact_goals out of i goals
                prob_player_exact = stats.binom.pmf(exact_goals, i, player_ratio)
                
                # Add to total probability
                total_prob += prob_team_i_goals * prob_player_exact
                
            return total_prob
    
    def predict_player_over_under(self, player_mean, threshold=0.5, score_grid=None, is_home_team=True, team_mean=None):
        """
        Calculate the probability of a player scoring over or under a threshold.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals (mean) for the player
        threshold : float
            Goal threshold
        score_grid : np.array, optional
            Pre-calculated scoreline probability grid. If provided, used instead of recalculating.
        is_home_team : bool
            Whether the player is on the home team (True) or away team (False)
        team_mean : float, optional
            Expected goals (mean) for the player's team. If None, player_mean is used directly.
            
        Returns:
        --------
        dict : Dictionary with probabilities for over and under
        """
        if team_mean is None or score_grid is None:
            # Use simple Poisson for individual player scores
            under_prob = stats.poisson.cdf(threshold - 0.5, player_mean)
            over_prob = 1 - under_prob
        
            return {
                'over': over_prob,
                'under': under_prob
            }
    
        # If team_mean provided with score_grid, use more accurate calculation
        if team_mean <= 0:
            return {
                'over': 0.0,
                'under': 1.0
            }
            
        # Player's contribution ratio
        player_ratio = player_mean / team_mean
        
        # Calculate exact goal probabilities from 0 to ceiling(threshold)
        threshold_ceil = int(np.ceil(threshold))
        exact_probs = np.zeros(threshold_ceil + 1)
        
        for goals in range(threshold_ceil + 1):
            exact_probs[goals] = self.predict_player_exact_goals(
                player_mean, team_mean, goals, score_grid, is_home_team
            )
            
        # Calculate under probability as sum of exact probabilities up to threshold
        under_prob = np.sum(exact_probs[:int(threshold + 0.5)])
        
        # Calculate over probability as 1 - under
        over_prob = 1 - under_prob
        
        return {
            'over': over_prob,
            'under': under_prob
        }
    
    def predict_player_scorer(self, player_mean, team_mean=None, score_grid=None, is_home_team=True):
        """
        Calculate the probability of a player scoring at least one goal.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals (mean) for the player
        team_mean : float, optional
            Expected goals (mean) for the player's team. If None, player_mean is used directly.
        score_grid : np.array, optional
            Pre-calculated scoreline probability grid. If provided, used instead of recalculating.
        is_home_team : bool
            Whether the player is on the home team (True) or away team (False)
            
        Returns:
        --------
        float : Probability of the player scoring at least one goal
        """
        if team_mean is None:
            # Simple Poisson calculation if team_mean not provided
            prob = 1 - stats.poisson.pmf(0, player_mean)
            return prob
        
        if team_mean <= 0:
            return 0
            
        # Player's contribution ratio
        player_ratio = player_mean / team_mean
        
        # If score grid is provided, use vectorized calculation
        if score_grid is not None:
            # Get the shape of the grid
            max_goals = score_grid.shape[0] - 1
            
            # Create coordinate matrices for team goals
            # Select rows/columns based on whether player is on home/away team
            if is_home_team:
                team_goals_coords = np.arange(max_goals + 1)[:, np.newaxis]  # Column vector
                grid_to_use = score_grid  # Use grid as is
            else:
                team_goals_coords = np.arange(max_goals + 1)[np.newaxis, :]  # Row vector
                grid_to_use = score_grid.T  # Transpose for away team
            
            # Create mask for non-zero team goals
            team_goals_mask = team_goals_coords > 0
            
            # Calculate probability of scoring at least once for each non-zero team score
            # 1 - (1 - player_ratio)^team_goals
            player_scores_probs = np.zeros_like(team_goals_coords, dtype=float)
            player_scores_probs[team_goals_mask] = 1 - np.power(1 - player_ratio, team_goals_coords[team_goals_mask])
            
            if is_home_team:
                # For home player: multiply each row by corresponding team goals probability
                # Reshape to broadcast across away goals
                scores_by_scoreline = grid_to_use * player_scores_probs
            else:
                # For away player: multiply each column by corresponding team goals probability
                # Reshape to broadcast across home goals
                scores_by_scoreline = grid_to_use * player_scores_probs
            
            # Sum all probabilities to get total scoring probability
            return np.sum(scores_by_scoreline)
        
        # If no score grid provided, use vectorized calculation with Poisson
        # Create an array of possible team goal values
        team_goals = np.arange(1, self.max_goals + 1)
        
        # Calculate probability of team scoring each goal count (Poisson)
        team_goals_probs = stats.poisson.pmf(team_goals, team_mean)
        
        # Calculate probability of player scoring at least once for each team score
        # 1 - (1 - player_ratio)^team_goals
        player_scores_probs = 1 - np.power(1 - player_ratio, team_goals)
        
        # Multiply and sum to get total scoring probability
        total_prob = np.sum(team_goals_probs * player_scores_probs)
            
        return total_prob
    
    def predict_player_2plus(self, player_mean, team_mean=None, score_grid=None, is_home_team=True):
        """
        Calculate the probability of a player scoring 2 or more goals.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals (mean) for the player
        team_mean : float, optional
            Expected goals (mean) for the player's team. If None, player_mean is used directly.
        score_grid : np.array, optional
            Pre-calculated scoreline probability grid. If provided, used instead of recalculating.
        is_home_team : bool
            Whether the player is on the home team (True) or away team (False)
            
        Returns:
        --------
        float : Probability of the player scoring 2+ goals
        """
        # First calculate anytime scorer probability as an upper bound
        anytime_scorer_prob = self.predict_player_scorer(player_mean, team_mean, score_grid, is_home_team)
        
        if team_mean is None:
            # Simple Poisson calculation if team_mean not provided
            prob = 1 - stats.poisson.pmf(0, player_mean) - stats.poisson.pmf(1, player_mean)
            return min(prob, anytime_scorer_prob)  # Ensure 2+ never exceeds anytime scorer
            
        if team_mean <= 0:
            return 0
            
        # Player's contribution ratio
        player_ratio = player_mean / team_mean
        
        # If score grid is provided, use vectorized calculation
        if score_grid is not None:
            # Get the shape of the grid
            max_goals = score_grid.shape[0] - 1
            
            # Total probability of scoring 2+ goals
            total_prob = 0
            
            # Create matrix of team goal counts
            if is_home_team:
                team_goals_matrix = np.arange(max_goals + 1)[:, np.newaxis]  # Column vector
                grid_to_use = score_grid
            else:
                team_goals_matrix = np.arange(max_goals + 1)[np.newaxis, :]  # Row vector
                grid_to_use = score_grid.T
            
            # Create mask for team goals ≥ 2
            team_goals_mask = team_goals_matrix >= 2
            
            # For each possible player goal count ≥ 2
            for player_goals in range(2, max_goals + 1):
                # Create mask where team goals ≥ player goals
                valid_mask = team_goals_matrix >= player_goals
                
                # Combined mask: team scores ≥ 2 AND team scores ≥ player goals
                combined_mask = np.logical_and(team_goals_mask, valid_mask)
                
                if np.any(combined_mask):
                    # Calculate binomial probability for exactly player_goals
                    # using the binomial PMF: binom(team_goals, player_goals, player_ratio)
                    player_goals_matrix = np.zeros_like(team_goals_matrix, dtype=float)
                    team_goals_values = team_goals_matrix[combined_mask]
                    
                    player_goals_matrix[combined_mask] = stats.binom.pmf(
                        player_goals,
                        team_goals_values,
                        player_ratio
                    )
                    
                    # Multiply by scoreline probabilities and add to total
                    if is_home_team:
                        total_prob += np.sum(grid_to_use * player_goals_matrix)
                    else:
                        total_prob += np.sum(grid_to_use * player_goals_matrix)
            
            return min(total_prob, anytime_scorer_prob)
        
        # If no score grid provided, use vectorized calculation with Poisson
        # Create an array of possible team goal values (starting from 2)
        team_goals = np.arange(2, self.max_goals + 1)
        
        # Calculate probability of team scoring each goal count
        team_goals_probs = stats.poisson.pmf(team_goals, team_mean)
        
        # Initialize matrix to store 2+ goal probabilities for each team score
        player_2plus_probs = np.zeros_like(team_goals, dtype=float)
        
        # For each team goal count, calculate probability of player scoring 2+ goals
        for i, goals in enumerate(team_goals):
            # Sum probabilities for player scoring 2, 3, ..., up to team_goals
            player_goal_counts = np.arange(2, goals + 1)
            binomial_probs = stats.binom.pmf(player_goal_counts, goals, player_ratio)
            player_2plus_probs[i] = np.sum(binomial_probs)
        
        # Multiply team goal probabilities by player 2+ goal probabilities and sum
        total_prob = np.sum(team_goals_probs * player_2plus_probs)
        
        # Ensure probability doesn't exceed anytime scorer probability
        return min(total_prob, anytime_scorer_prob)
    
    def predict_player_3plus(self, player_mean, team_mean=None, score_grid=None, is_home_team=True):
        """
        Calculate the probability of a player scoring 3 or more goals.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals (mean) for the player
        team_mean : float, optional
            Expected goals (mean) for the player's team. If None, player_mean is used directly.
        score_grid : np.array, optional
            Pre-calculated scoreline probability grid. If provided, used instead of recalculating.
        is_home_team : bool
            Whether the player is on the home team (True) or away team (False)
            
        Returns:
        --------
        float : Probability of the player scoring 3+ goals
        """
        # First calculate anytime scorer probability as an upper bound
        anytime_scorer_prob = self.predict_player_scorer(player_mean, team_mean, score_grid, is_home_team)
        
        if team_mean is None:
            # Simple Poisson calculation if team_mean not provided
            prob = 1 - stats.poisson.pmf(0, player_mean) - stats.poisson.pmf(1, player_mean) - stats.poisson.pmf(2, player_mean)
            return min(prob, anytime_scorer_prob)  # Ensure 3+ never exceeds anytime scorer
            
        if team_mean <= 0:
            return 0
            
        # Player's contribution ratio
        player_ratio = player_mean / team_mean
        
        # If score grid is provided, use vectorized calculation
        if score_grid is not None:
            # Get the shape of the grid
            max_goals = score_grid.shape[0] - 1
            
            # Total probability of scoring 3+ goals
            total_prob = 0
            
            # Create matrix of team goal counts
            if is_home_team:
                team_goals_matrix = np.arange(max_goals + 1)[:, np.newaxis]  # Column vector
                grid_to_use = score_grid
            else:
                team_goals_matrix = np.arange(max_goals + 1)[np.newaxis, :]  # Row vector
                grid_to_use = score_grid.T
            
            # Create mask for team goals ≥ 3
            team_goals_mask = team_goals_matrix >= 3
            
            # For each possible player goal count ≥ 3
            for player_goals in range(3, max_goals + 1):
                # Create mask where team goals ≥ player goals
                valid_mask = team_goals_matrix >= player_goals
                
                # Combined mask: team scores ≥ 3 AND team scores ≥ player goals
                combined_mask = np.logical_and(team_goals_mask, valid_mask)
                
                if np.any(combined_mask):
                    # Calculate binomial probability for exactly player_goals
                    # using the binomial PMF: binom(team_goals, player_goals, player_ratio)
                    player_goals_matrix = np.zeros_like(team_goals_matrix, dtype=float)
                    team_goals_values = team_goals_matrix[combined_mask]
                    
                    player_goals_matrix[combined_mask] = stats.binom.pmf(
                        player_goals,
                        team_goals_values,
                        player_ratio
                    )
                    
                    # Multiply by scoreline probabilities and add to total
                    if is_home_team:
                        total_prob += np.sum(grid_to_use * player_goals_matrix)
                    else:
                        total_prob += np.sum(grid_to_use * player_goals_matrix)
            
            return min(total_prob, anytime_scorer_prob)
        
        # If no score grid provided, use vectorized calculation with Poisson
        # Create an array of possible team goal values (starting from 3)
        team_goals = np.arange(3, self.max_goals + 1)
        
        # Calculate probability of team scoring each goal count
        team_goals_probs = stats.poisson.pmf(team_goals, team_mean)
        
        # Initialize matrix to store 3+ goal probabilities for each team score
        player_3plus_probs = np.zeros_like(team_goals, dtype=float)
        
        # For each team goal count, calculate probability of player scoring 3+ goals
        for i, goals in enumerate(team_goals):
            # Sum probabilities for player scoring 3, 4, ..., up to team_goals
            player_goal_counts = np.arange(3, goals + 1)
            binomial_probs = stats.binom.pmf(player_goal_counts, goals, player_ratio)
            player_3plus_probs[i] = np.sum(binomial_probs)
        
        # Multiply team goal probabilities by player 3+ goal probabilities and sum
        total_prob = np.sum(team_goals_probs * player_3plus_probs)
        
        # Ensure probability doesn't exceed anytime scorer probability
        return min(total_prob, anytime_scorer_prob)
    
    def predict_player_scorer_with_btts_and_ou(self, player_mean, team_mean, home_mean, away_mean, 
                                      threshold=2.5, is_home_team=True, rho=-0.16, score_grid=None):
        """
        Calculate combined probability for a player to score and BTTS and over/under
        
        Parameters:
        -----------
        player_mean : float
            Expected goals for the player
        team_mean : float
            Expected goals for the player's team
        home_mean, away_mean : float
            Expected goals for home and away teams
        threshold : float
            Threshold for over/under (default: 2.5)
        is_home_team : bool
            True if player is on home team, False if on away team
        rho : float
            Correlation parameter
        score_grid : np.array, optional
            Pre-calculated score probability grid
            
        Returns:
        --------
        dict : Dictionary with probabilities for different combinations
        """
        if team_mean <= 0:
            return {
                'yes_over': 0,
                'yes_under': 0,
                'no_over': 0,
                'no_under': 0
            }
        
        # Player's contribution ratio to team goals
        player_ratio = player_mean / team_mean
        
        # Get or calculate the score grid
        if score_grid is None:
            grid = self.predict_score_grid(home_mean, away_mean, self.max_goals, rho)
            max_goals = self.max_goals
        else:
            grid = score_grid
            max_goals = min(grid.shape[0] - 1, grid.shape[1] - 1)
        
        # Create coordinate matrices
        home_goals, away_goals = np.meshgrid(
            np.arange(max_goals + 1),
            np.arange(max_goals + 1),
            indexing='ij'
        )
        
        # Create total goals matrix
        total_goals = home_goals + away_goals
        
        # Set up team goals matrix based on which team the player is on
        if is_home_team:
            team_goals = home_goals
            opponent_goals = away_goals
        else:
            team_goals = away_goals
            opponent_goals = home_goals
        
        # Create BTTS mask: home_goals > 0 AND away_goals > 0
        btts_mask = (home_goals > 0) & (away_goals > 0)
        no_btts_mask = ~btts_mask
        
        # Create over/under masks
        over_mask = total_goals >= threshold
        under_mask = ~over_mask
        
        # Create player scoring probability matrix
        # First, create a mask for non-zero team goals (player can only score if team scores)
        team_scores_mask = team_goals > 0
        
        # Initialize player scoring probability matrix
        player_scores_prob = np.zeros_like(grid)
        
        # For each team goal count > 0, calculate probability player scores at least once
        if np.any(team_scores_mask):
            # Get the team goal counts where team scored
            valid_team_goals = team_goals[team_scores_mask]
            
            # Calculate probability player scores at least once: 1 - (1 - player_ratio) ^ team_goals
            player_scores_prob[team_scores_mask] = 1 - np.power(1 - player_ratio, valid_team_goals)
        
        # Calculate joint probabilities: scoreline probability * player scoring probability
        joint_probs = grid * player_scores_prob
        
        # Calculate probabilities for each combination using masks
        yes_over_mask = btts_mask & over_mask
        yes_under_mask = btts_mask & under_mask
        no_over_mask = no_btts_mask & over_mask
        no_under_mask = no_btts_mask & under_mask
        
        # Sum probabilities for each combination
        yes_over = np.sum(joint_probs[yes_over_mask])
        yes_under = np.sum(joint_probs[yes_under_mask])
        no_over = np.sum(joint_probs[no_over_mask])
        no_under = np.sum(joint_probs[no_under_mask])
        
        return {
            'yes_over': yes_over,
            'yes_under': yes_under,
            'no_over': no_over,
            'no_under': no_under
        }
        
    def predict_player_scorer_with_nplus_and_result(self, player_mean, team_mean, home_mean, away_mean, 
                                                  is_home_team=True, rho=-0.16, n_plus=2, score_grid=None):
        """
        Calculate probabilities for player scoring n+ goals combined with match result.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals (mean) for the player
        team_mean : float
            Expected goals (mean) for the player's team (either home or away)
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        is_home_team : bool
            True if player is on the home team, False if on away team
        rho : float
            Correlation parameter for low-scoring adjustment
        n_plus : int
            Number of goals for the player to score (2 for 2+, 3 for 3+)
        score_grid : np.array, optional
            Pre-calculated scoreline probability grid
            
        Returns:
        --------
        dict : Dictionary with probabilities for different combinations
        """
        if team_mean <= 0:
            return {
                'scores_team_wins': 0,
                'scores_team_draws': 0,
                'scores_team_loses': 0,
                'no_score_team_wins': 0,
                'no_score_team_draws': 0,
                'no_score_team_loses': 0
            }
        
        # Player's contribution ratio to team goals
        player_ratio = player_mean / team_mean
        
        # Get or calculate the score grid
        if score_grid is None:
            grid = self.predict_score_grid(home_mean, away_mean, self.max_goals, rho)
            max_goals = self.max_goals
        else:
            grid = score_grid
            max_goals = min(grid.shape[0] - 1, grid.shape[1] - 1)
        
        # Create coordinate matrices
        home_goals, away_goals = np.meshgrid(
            np.arange(max_goals + 1),
            np.arange(max_goals + 1),
            indexing='ij'
        )
        
        # Set up team goals matrix based on which team the player is on
        if is_home_team:
            team_goals = home_goals
        else:
            team_goals = away_goals
            
        # Create match result masks
        if is_home_team:
            team_wins_mask = home_goals > away_goals
            team_draws_mask = home_goals == away_goals
            team_loses_mask = home_goals < away_goals
        else:
            team_wins_mask = away_goals > home_goals
            team_draws_mask = home_goals == away_goals
            team_loses_mask = away_goals < home_goals
            
        # Create mask for team goals >= n_plus (minimum required for player to score n+ goals)
        nplus_possible_mask = team_goals >= n_plus
        
        # Create a matrix to store player n+ goal probabilities for each scoreline
        player_nplus_probs = np.zeros_like(grid)
        
        # For each valid team goal count (>= n_plus), calculate n+ probability
        valid_indices = np.where(nplus_possible_mask)
        
        # Initialize an array to store all the scores
        all_scores = team_goals[valid_indices]
        
        # Calculate player n+ probabilities for all valid scorelines at once
        all_probs = np.zeros_like(all_scores, dtype=float)
        
        # Calculate probability of scoring AT LEAST n goals
        # This is 1 minus probability of scoring less than n goals
        for i, team_score in enumerate(all_scores):
            prob_less_than_n = 0
            for k in range(n_plus):
                prob_less_than_n += stats.binom.pmf(k, team_score, player_ratio)
            all_probs[i] = 1 - prob_less_than_n
            
        # Assign calculated probabilities back to the matrix
        player_nplus_probs[valid_indices] = all_probs
        
        # Multiply by the score grid to get joint probabilities
        joint_probs = grid * player_nplus_probs
        
        # Calculate outcome probabilities
        scores_team_wins = np.sum(joint_probs[team_wins_mask])
        scores_team_draws = np.sum(joint_probs[team_draws_mask])
        scores_team_loses = np.sum(joint_probs[team_loses_mask])
        
        # Calculate no-score outcome probabilities
        no_score_team_wins = np.sum(grid[team_wins_mask]) - scores_team_wins
        no_score_team_draws = np.sum(grid[team_draws_mask]) - scores_team_draws
        no_score_team_loses = np.sum(grid[team_loses_mask]) - scores_team_loses
        
        return {
            'scores_team_wins': scores_team_wins,
            'scores_team_draws': scores_team_draws,
            'scores_team_loses': scores_team_loses,
            'no_score_team_wins': no_score_team_wins,
            'no_score_team_draws': no_score_team_draws,
            'no_score_team_loses': no_score_team_loses
        }

    def predict_btts_and_match_result(self, home_mean, away_mean, rho, score_grid=None):
        """
        Calculate probabilities for Both Teams To Score (BTTS) combined with match result.
        
        Parameters:
        -----------
        home_mean, away_mean : float
            Expected goals (mean) for home and away teams
        rho : float
            Correlation parameter for low-scoring adjustment
        score_grid : np.array, optional
            Pre-calculated score probability grid. If None, will calculate it.
            
        Returns:
        --------
        dict : Dictionary containing probabilities for each BTTS & match result combination
        """
        if score_grid is None:
            score_grid = self.predict_score_grid(home_mean, away_mean, rho=rho)
            
        max_goals = score_grid.shape[0] - 1
        
        # Initialize probabilities
        btts_home_win = 0.0
        btts_draw = 0.0
        btts_away_win = 0.0
        no_btts_home_win = 0.0
        no_btts_draw = 0.0
        no_btts_away_win = 0.0
        
        # Calculate probabilities for each scoreline
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = score_grid[i, j]
                
                # Both teams score
                if i > 0 and j > 0:
                    if i > j:  # Home win
                        btts_home_win += prob
                    elif i < j:  # Away win
                        btts_away_win += prob
                    else:  # Draw
                        btts_draw += prob
                # Not both teams score
                else:
                    if i > j:  # Home win
                        no_btts_home_win += prob
                    elif i < j:  # Away win
                        no_btts_away_win += prob
                    else:  # Draw
                        no_btts_draw += prob
        
        return {
            'yes_home': btts_home_win,
            'yes_draw': btts_draw,
            'yes_away': btts_away_win,
            'no_home': no_btts_home_win,
            'no_draw': no_btts_draw,
            'no_away': no_btts_away_win
        }

    def predict_player_scorer_with_match_over_under(self, player_mean, team_mean, home_mean, away_mean, threshold, is_home_team, rho, score_grid=None):
        """
        Calculate probabilities for player scoring combined with match over/under.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals for the player
        team_mean : float
            Expected goals for the player's team
        home_mean, away_mean : float
            Expected goals for home and away teams
        threshold : float
            Over/under threshold (e.g., 2.5)
        is_home_team : bool
            Whether the player is on the home team
        rho : float
            Correlation parameter for low-scoring adjustment
        score_grid : np.array, optional
            Pre-calculated score probability grid. If None, will calculate it.
            
        Returns:
        --------
        dict : Dictionary containing probabilities for each combination
        """
        if score_grid is None:
            score_grid = self.predict_score_grid(home_mean, away_mean, rho=rho)
            
        max_goals = score_grid.shape[0] - 1
        
        # Initialize probabilities
        scores_over = 0.0
        scores_under = 0.0
        no_score_over = 0.0
        no_score_under = 0.0
        
        # Calculate probabilities for each scoreline
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = score_grid[i, j]
                total_goals = i + j
                
                # Calculate probability of player scoring given the scoreline
                if is_home_team:
                    team_goals = i
                else:
                    team_goals = j
                    
                # Calculate probability of player scoring at least one goal
                if team_goals > 0:
                    # Use binomial probability for player scoring at least one goal
                    p_score = 1 - (1 - player_mean/team_mean) ** team_goals
                    p_no_score = 1 - p_score
                else:
                    p_score = 0
                    p_no_score = 1
                
                # Combine with over/under
                if total_goals > threshold:
                    scores_over += prob * p_score
                    no_score_over += prob * p_no_score
                else:
                    scores_under += prob * p_score
                    no_score_under += prob * p_no_score
        
        return {
            'scores_over': scores_over,
            'scores_under': scores_under,
            'no_score_over': no_score_over,
            'no_score_under': no_score_under
        }

    def predict_player_scorer_with_home_team_over_under(self, player_mean, team_mean, home_mean, away_mean, threshold, is_home_team, rho, score_grid=None):
        """
        Calculate probabilities for player scoring combined with home team over/under.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals for the player
        team_mean : float
            Expected goals for the player's team
        home_mean, away_mean : float
            Expected goals for home and away teams
        threshold : float
            Over/under threshold (e.g., 2.5)
        is_home_team : bool
            Whether the player is on the home team
        rho : float
            Correlation parameter for low-scoring adjustment
        score_grid : np.array, optional
            Pre-calculated score probability grid. If None, will calculate it.
            
        Returns:
        --------
        dict : Dictionary containing probabilities for each combination
        """
        if score_grid is None:
            score_grid = self.predict_score_grid(home_mean, away_mean, rho=rho)
            
        max_goals = score_grid.shape[0] - 1
        
        # Initialize probabilities
        scores_over = 0.0
        scores_under = 0.0
        no_score_over = 0.0
        no_score_under = 0.0
        
        # Calculate probabilities for each scoreline
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = score_grid[i, j]
                
                # Calculate probability of player scoring given the scoreline
                if is_home_team:
                    team_goals = i
                else:
                    team_goals = j
                    
                # Calculate probability of player scoring at least one goal
                if team_goals > 0:
                    # Use binomial probability for player scoring at least one goal
                    p_score = 1 - (1 - player_mean/team_mean) ** team_goals
                    p_no_score = 1 - p_score
                else:
                    p_score = 0
                    p_no_score = 1
                
                # Combine with home team over/under
                if i > threshold:
                    scores_over += prob * p_score
                    no_score_over += prob * p_no_score
                else:
                    scores_under += prob * p_score
                    no_score_under += prob * p_no_score
        
        return {
            'scores_over': scores_over,
            'scores_under': scores_under,
            'no_score_over': no_score_over,
            'no_score_under': no_score_under
        }
        
    def predict_player_scorer_with_away_team_over_under(self, player_mean, team_mean, home_mean, away_mean, threshold, is_home_team, rho, score_grid=None):
        """
        Calculate probabilities for player scoring combined with away team over/under.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals for the player
        team_mean : float
            Expected goals for the player's team
        home_mean, away_mean : float
            Expected goals for home and away teams
        threshold : float
            Over/under threshold (e.g., 2.5)
        is_home_team : bool
            Whether the player is on the home team
        rho : float
            Correlation parameter for low-scoring adjustment
        score_grid : np.array, optional
            Pre-calculated score probability grid. If None, will calculate it.
            
        Returns:
        --------
        dict : Dictionary containing probabilities for each combination
        """
        if score_grid is None:
            score_grid = self.predict_score_grid(home_mean, away_mean, rho=rho)
            
        max_goals = score_grid.shape[0] - 1
        
        # Initialize probabilities
        scores_over = 0.0
        scores_under = 0.0
        no_score_over = 0.0
        no_score_under = 0.0
        
        # Calculate probabilities for each scoreline
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = score_grid[i, j]
                
                # Calculate probability of player scoring given the scoreline
                if is_home_team:
                    team_goals = i
                else:
                    team_goals = j
                    
                # Calculate probability of player scoring at least one goal
                if team_goals > 0:
                    # Use binomial probability for player scoring at least one goal
                    p_score = 1 - (1 - player_mean/team_mean) ** team_goals
                    p_no_score = 1 - p_score
                else:
                    p_score = 0
                    p_no_score = 1
                
                # Combine with away team over/under
                if j > threshold:
                    scores_over += prob * p_score
                    no_score_over += prob * p_no_score
                else:
                    scores_under += prob * p_score
                    no_score_under += prob * p_no_score
        
        return {
            'scores_over': scores_over,
            'scores_under': scores_under,
            'no_score_over': no_score_over,
            'no_score_under': no_score_under
        }

    def predict_player_scorer_with_btts_and_match_result(self, player_mean, team_mean, home_mean, away_mean, is_home_team, rho, score_grid=None):
        """
        Calculate probabilities for player scoring combined with BTTS and match result.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals for the player
        team_mean : float
            Expected goals for the player's team
        home_mean, away_mean : float
            Expected goals for home and away teams
        is_home_team : bool
            Whether the player is on the home team
        rho : float
            Correlation parameter for low-scoring adjustment
        score_grid : np.array, optional
            Pre-calculated score probability grid. If None, will calculate it.
            
        Returns:
        --------
        dict : Dictionary containing probabilities for each combination
        """
        if score_grid is None:
            score_grid = self.predict_score_grid(home_mean, away_mean, rho=rho)
            
        max_goals = score_grid.shape[0] - 1
        
        # Initialize probabilities
        scores_yes_home = 0.0
        scores_yes_draw = 0.0
        scores_yes_away = 0.0
        no_score_yes_home = 0.0
        no_score_yes_draw = 0.0
        no_score_yes_away = 0.0
        scores_no_home = 0.0
        scores_no_draw = 0.0
        scores_no_away = 0.0
        no_score_no_home = 0.0
        no_score_no_draw = 0.0
        no_score_no_away = 0.0
        
        # Calculate probabilities for each scoreline
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = score_grid[i, j]
                
                # Calculate probability of player scoring given the scoreline
                if is_home_team:
                    team_goals = i
                else:
                    team_goals = j
                    
                # Calculate probability of player scoring at least one goal
                if team_goals > 0:
                    # Use binomial probability for player scoring at least one goal
                    p_score = 1 - (1 - player_mean/team_mean) ** team_goals
                    p_no_score = 1 - p_score
                else:
                    p_score = 0
                    p_no_score = 1
                
                # Determine BTTS and match result
                btts = i > 0 and j > 0
                
                if i > j:  # Home win
                    if btts:
                        scores_yes_home += prob * p_score
                        no_score_yes_home += prob * p_no_score
                    else:
                        scores_no_home += prob * p_score
                        no_score_no_home += prob * p_no_score
                elif i < j:  # Away win
                    if btts:
                        scores_yes_away += prob * p_score
                        no_score_yes_away += prob * p_no_score
                    else:
                        scores_no_away += prob * p_score
                        no_score_no_away += prob * p_no_score
                else:  # Draw
                    if btts:
                        scores_yes_draw += prob * p_score
                        no_score_yes_draw += prob * p_no_score
                    else:
                        scores_no_draw += prob * p_score
                        no_score_no_draw += prob * p_no_score
        
        return {
            'scores_yes_home': scores_yes_home,
            'scores_yes_draw': scores_yes_draw,
            'scores_yes_away': scores_yes_away,
            'no_score_yes_home': no_score_yes_home,
            'no_score_yes_draw': no_score_yes_draw,
            'no_score_yes_away': no_score_yes_away,
            'scores_no_home': scores_no_home,
            'scores_no_draw': scores_no_draw,
            'scores_no_away': scores_no_away,
            'no_score_no_home': no_score_no_home,
            'no_score_no_draw': no_score_no_draw,
            'no_score_no_away': no_score_no_away
        }

    def predict_player_scorer_with_nplus_btts_and_ou(self, player_mean, team_mean, home_mean, away_mean, 
                                      threshold=2.5, is_home_team=True, rho=-0.16, n_plus=2, score_grid=None):
        """
        Calculate combined probability for a player to score n+ goals and BTTS and over/under
        
        Parameters:
        -----------
        player_mean : float
            Expected goals for the player
        team_mean : float
            Expected goals for the player's team
        home_mean, away_mean : float
            Expected goals for home and away teams
        threshold : float
            Threshold for over/under (default: 2.5)
        is_home_team : bool
            True if player is on home team, False if on away team
        rho : float
            Correlation parameter
        n_plus : int
            Number of goals for the player to score (2 for 2+, 3 for 3+)
        score_grid : np.array, optional
            Pre-calculated scoreline probability grid
            
        Returns:
        --------
        dict : Dictionary with probabilities for different combinations
        """
        if team_mean <= 0:
            return {
                'yes_over': 0,
                'yes_under': 0,
                'no_over': 0,
                'no_under': 0
            }
        
        # Player's contribution ratio to team goals
        player_ratio = player_mean / team_mean
        
        # Get or calculate the score grid
        if score_grid is None:
            grid = self.predict_score_grid(home_mean, away_mean, self.max_goals, rho)
            max_goals = self.max_goals
        else:
            grid = score_grid
            max_goals = min(grid.shape[0] - 1, grid.shape[1] - 1)
        
        # Create coordinate matrices
        home_goals, away_goals = np.meshgrid(
            np.arange(max_goals + 1),
            np.arange(max_goals + 1),
            indexing='ij'
        )
        
        # Create total goals matrix
        total_goals = home_goals + away_goals
        
        # Set up team goals matrix based on which team the player is on
        if is_home_team:
            team_goals = home_goals
            opponent_goals = away_goals
        else:
            team_goals = away_goals
            opponent_goals = home_goals
        
        # Create BTTS mask: home_goals > 0 AND away_goals > 0
        btts_mask = (home_goals > 0) & (away_goals > 0)
        no_btts_mask = ~btts_mask
        
        # Create over/under masks
        over_mask = total_goals >= threshold
        under_mask = ~over_mask
        
        # Create mask for team goals >= n_plus (minimum required for player to score n+ goals)
        nplus_possible_mask = team_goals >= n_plus
        
        # Create a matrix to store player n+ goal probabilities for each scoreline
        player_nplus_probs = np.zeros_like(grid)
        
        # For each valid team goal count (>= n_plus), calculate n+ probability
        valid_indices = np.where(nplus_possible_mask)
        
        # Initialize an array to store all the scores
        all_scores = team_goals[valid_indices]
        
        # Calculate player n+ probabilities for all valid scorelines at once
        all_probs = np.zeros_like(all_scores, dtype=float)
        
        # Calculate probability of scoring AT LEAST n goals
        # This is 1 minus probability of scoring less than n goals
        for i, team_score in enumerate(all_scores):
            prob_less_than_n = 0
            for k in range(n_plus):
                prob_less_than_n += stats.binom.pmf(k, team_score, player_ratio)
            all_probs[i] = 1 - prob_less_than_n
            
        # Assign calculated probabilities back to the matrix
        player_nplus_probs[valid_indices] = all_probs
        
        # Calculate joint probabilities: scoreline probability * player scoring probability
        joint_probs = grid * player_nplus_probs
        
        # Calculate probabilities for each combination using masks
        yes_over_mask = btts_mask & over_mask
        yes_under_mask = btts_mask & under_mask
        no_over_mask = no_btts_mask & over_mask
        no_under_mask = no_btts_mask & under_mask
        
        # Sum probabilities for each combination
        yes_over = np.sum(joint_probs[yes_over_mask])
        yes_under = np.sum(joint_probs[yes_under_mask])
        no_over = np.sum(joint_probs[no_over_mask])
        no_under = np.sum(joint_probs[no_under_mask])
        
        return {
            'yes_over': yes_over,
            'yes_under': yes_under,
            'no_over': no_over,
            'no_under': no_under
        }

    def predict_player_scorer_with_nplus_btts_and_match_result(self, player_mean, team_mean, home_mean, away_mean, is_home_team, rho, n_plus=2, score_grid=None):
        """
        Calculate probabilities for player scoring n+ goals combined with BTTS and match result.
        
        Parameters:
        -----------
        player_mean : float
            Expected goals for the player
        team_mean : float
            Expected goals for the player's team
        home_mean, away_mean : float
            Expected goals for home and away teams
        is_home_team : bool
            Whether the player is on the home team
        rho : float
            Correlation parameter for low-scoring adjustment
        n_plus : int
            Number of goals for the player to score (2 for 2+, 3 for 3+)
        score_grid : np.array, optional
            Pre-calculated score probability grid. If None, will calculate it.
            
        Returns:
        --------
        dict : Dictionary containing probabilities for each combination
        """
        if score_grid is None:
            score_grid = self.predict_score_grid(home_mean, away_mean, rho=rho)
            
        max_goals = score_grid.shape[0] - 1
        
        # Initialize probabilities
        scores_yes_home = 0.0
        scores_yes_draw = 0.0
        scores_yes_away = 0.0
        no_score_yes_home = 0.0
        no_score_yes_draw = 0.0
        no_score_yes_away = 0.0
        scores_no_home = 0.0
        scores_no_draw = 0.0
        scores_no_away = 0.0
        no_score_no_home = 0.0
        no_score_no_draw = 0.0
        no_score_no_away = 0.0
        
        # Player's contribution ratio to team goals
        player_ratio = player_mean / team_mean
        
        # Create coordinate matrices
        home_goals, away_goals = np.meshgrid(
            np.arange(max_goals + 1),
            np.arange(max_goals + 1),
            indexing='ij'
        )
        
        # Set up team goals matrix based on which team the player is on
        if is_home_team:
            team_goals = home_goals
        else:
            team_goals = away_goals
        
        # Create mask for team goals >= n_plus (minimum required for player to score n+ goals)
        nplus_possible_mask = team_goals >= n_plus
        
        # Create a matrix to store player n+ goal probabilities for each scoreline
        player_nplus_probs = np.zeros_like(score_grid)
        
        # For each valid team goal count (>= n_plus), calculate n+ probability
        valid_indices = np.where(nplus_possible_mask)
        
        # Initialize an array to store all the scores
        all_scores = team_goals[valid_indices]
        
        # Calculate player n+ probabilities for all valid scorelines at once
        all_probs = np.zeros_like(all_scores, dtype=float)
        
        # Calculate probability of scoring AT LEAST n goals
        # This is 1 minus probability of scoring less than n goals
        for i, team_score in enumerate(all_scores):
            prob_less_than_n = 0
            for k in range(n_plus):
                prob_less_than_n += stats.binom.pmf(k, team_score, player_ratio)
            all_probs[i] = 1 - prob_less_than_n
            
        # Assign calculated probabilities back to the matrix
        player_nplus_probs[valid_indices] = all_probs
        
        # Calculate probabilities for each scoreline
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                prob = score_grid[i, j]
                
                # Calculate probability of player scoring n+ goals given the scoreline
                if is_home_team:
                    team_goals_value = i
                else:
                    team_goals_value = j
                    
                # Calculate probability of player scoring at least n goals
                if team_goals_value >= n_plus:
                    p_score = player_nplus_probs[i, j]
                    p_no_score = 1 - p_score
                else:
                    p_score = 0
                    p_no_score = 1
                
                # Determine BTTS and match result
                btts = i > 0 and j > 0
                
                if i > j:  # Home win
                    if btts:
                        scores_yes_home += prob * p_score
                        no_score_yes_home += prob * p_no_score
                    else:
                        scores_no_home += prob * p_score
                        no_score_no_home += prob * p_no_score
                elif i < j:  # Away win
                    if btts:
                        scores_yes_away += prob * p_score
                        no_score_yes_away += prob * p_no_score
                    else:
                        scores_no_away += prob * p_score
                        no_score_no_away += prob * p_no_score
                else:  # Draw
                    if btts:
                        scores_yes_draw += prob * p_score
                        no_score_yes_draw += prob * p_no_score
                    else:
                        scores_no_draw += prob * p_score
                        no_score_no_draw += prob * p_no_score
        
        return {
            'scores_yes_home': scores_yes_home,
            'scores_yes_draw': scores_yes_draw,
            'scores_yes_away': scores_yes_away,
            'no_score_yes_home': no_score_yes_home,
            'no_score_yes_draw': no_score_yes_draw,
            'no_score_yes_away': no_score_yes_away,
            'scores_no_home': scores_no_home,
            'scores_no_draw': scores_no_draw,
            'scores_no_away': scores_no_away,
            'no_score_no_home': no_score_no_home,
            'no_score_no_draw': no_score_no_draw,
            'no_score_no_away': no_score_no_away
        }