import sys
import numpy as np
from scipy import stats
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, 
    QSpinBox, QDoubleSpinBox, QTabWidget, QScrollArea,
    QGroupBox, QFormLayout, QMessageBox, QSizePolicy,
    QCheckBox, QRadioButton, QButtonGroup, QFileDialog,
    QApplication, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QSlider, QGridLayout, QDialog
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont, QColor, QPalette

from models.dixon_coles import DixonColesModel
from models.visualize import (
    MatplotlibCanvas, create_heatmap_for_canvas, 
    format_outcome_probabilities, format_over_under,
    create_player_scoring_heatmap_for_canvas
)


def probability_to_decimal_odds(prob):
    """Convert probability to decimal odds"""
    return 1 / prob if prob > 0 else float('inf')

def decimal_odds_to_probability(odds):
    """Convert decimal odds to probability"""
    return 1 / odds if odds > 0 else 0.0

def calculate_player_mean_from_odds(odds, team_mean, max_goals=10):
    """
    Calculate player expected goals (mean) from anytime scorer odds
    
    Parameters:
    -----------
    odds : float
        Decimal odds for the player to score anytime
    team_mean : float
        Expected goals for the player's team
    max_goals : int
        Maximum number of goals to consider
    
    Returns:
    --------
    float : Calculated player mean
    """
    if odds <= 1.0 or team_mean <= 0:
        return 0.0
        
    # Convert decimal odds to probability
    target_prob = decimal_odds_to_probability(odds)
    
    # Define a function to minimize
    def objective(player_mean):
        # Calculate player ratio
        player_ratio = player_mean / team_mean
        
        # Probability player doesn't score
        prob_not_score = 0
        
        # Calculate for each possible number of team goals (up to max_goals)
        for i in range(max_goals + 1):
            # Probability team scores i goals
            prob_team_i_goals = stats.poisson.pmf(i, team_mean)
            
            # Probability player doesn't score any of those i goals
            if i == 0:
                prob_player_not_score = 1  # If team doesn't score, player can't score
            else:
                prob_player_not_score = (1 - player_ratio) ** i
                
            # Add to total probability
            prob_not_score += prob_team_i_goals * prob_player_not_score
            
        # Calculate anytime scorer probability
        calculated_prob = 1 - prob_not_score
        
        # Return the absolute difference between calculated and target probabilities
        return abs(calculated_prob - target_prob)
    
    # Starting guess: a reasonable proportion of team's goals
    initial_guess = team_mean * 0.3
    
    # Use binary search to find the player_mean that gives the target probability
    lower_bound = 0.0
    upper_bound = team_mean * 2  # Player can't score more than twice the team mean (reasonable limit)
    
    # Maximum iterations to prevent infinite loops
    max_iterations = 50
    iterations = 0
    
    while iterations < max_iterations:
        mid = (lower_bound + upper_bound) / 2
        
        # Calculate scoring probability with current mid value
        player_ratio = mid / team_mean
        prob_not_score = 0
        
        for i in range(max_goals + 1):
            prob_team_i_goals = stats.poisson.pmf(i, team_mean)
            if i == 0:
                prob_player_not_score = 1
            else:
                prob_player_not_score = (1 - player_ratio) ** i
            prob_not_score += prob_team_i_goals * prob_player_not_score
        
        calculated_prob = 1 - prob_not_score
        
        # Check if we're close enough
        if abs(calculated_prob - target_prob) < 0.0001:
            return mid
            
        # Adjust bounds
        if calculated_prob < target_prob:
            lower_bound = mid
        else:
            upper_bound = mid
            
        iterations += 1
    
    # Return the best estimate after max iterations
    return (lower_bound + upper_bound) / 2


class BettingMarketWidget(QWidget):
    """
    Widget for displaying betting market odds
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the betting market display UI"""
        main_layout = QVBoxLayout(self)
        
        # 1X2 Market
        self.match_outcome_group = QGroupBox("Match Outcome (1X2)")
        match_outcome_layout = QHBoxLayout(self.match_outcome_group)
        
        self.home_win_btn = QPushButton()
        self.home_win_btn.setStyleSheet("""
            background-color: #2d7dd2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 16px 12px 16px;
            font-weight: bold;
            text-align: center;
            min-width: 150px;
            min-height: 60px;
            font-size: 11pt;
        """)
        
        self.draw_btn = QPushButton()
        self.draw_btn.setStyleSheet("""
            background-color: #2d7dd2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 16px 12px 16px;
            font-weight: bold;
            text-align: center;
            min-width: 150px;
            min-height: 60px;
            font-size: 11pt;
        """)
        
        self.away_win_btn = QPushButton()
        self.away_win_btn.setStyleSheet("""
            background-color: #2d7dd2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 16px 12px 16px;
            font-weight: bold;
            text-align: center;
            min-width: 150px;
            min-height: 60px;
            font-size: 11pt;
        """)
        
        match_outcome_layout.addWidget(self.home_win_btn)
        match_outcome_layout.addWidget(self.draw_btn)
        match_outcome_layout.addWidget(self.away_win_btn)
        
        main_layout.addWidget(self.match_outcome_group)
        
        # Over/Under Markets
        self.over_under_group = QGroupBox("Over/Under Markets")
        ou_layout = QGridLayout(self.over_under_group)
        
        # Labels
        ou_layout.addWidget(QLabel("Total Goals"), 0, 0)
        ou_layout.addWidget(QLabel("Home Team"), 0, 1)
        ou_layout.addWidget(QLabel("Away Team"), 0, 2)
        
        # Create over/under buttons for various thresholds
        self.ou_total_btns = {}
        self.ou_home_btns = {}
        self.ou_away_btns = {}
        
        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        
        for i, threshold in enumerate(thresholds):
            # Total goals
            total_layout = QHBoxLayout()
            over_btn = QPushButton(f"Over {threshold}")
            under_btn = QPushButton(f"Under {threshold}")
            self.ou_total_btns[threshold] = (over_btn, under_btn)
            total_layout.addWidget(over_btn)
            total_layout.addWidget(under_btn)
            ou_layout.addLayout(total_layout, i+1, 0)
            
            # Home team goals
            home_layout = QHBoxLayout()
            over_home_btn = QPushButton(f"Over {threshold}")
            under_home_btn = QPushButton(f"Under {threshold}")
            self.ou_home_btns[threshold] = (over_home_btn, under_home_btn)
            home_layout.addWidget(over_home_btn)
            home_layout.addWidget(under_home_btn)
            ou_layout.addLayout(home_layout, i+1, 1)
            
            # Away team goals
            away_layout = QHBoxLayout()
            over_away_btn = QPushButton(f"Over {threshold}")
            under_away_btn = QPushButton(f"Under {threshold}")
            self.ou_away_btns[threshold] = (over_away_btn, under_away_btn)
            away_layout.addWidget(over_away_btn)
            away_layout.addWidget(under_away_btn)
            ou_layout.addLayout(away_layout, i+1, 2)
        
        main_layout.addWidget(self.over_under_group)
        
        # Both Teams To Score (BTTS) Market
        self.btts_group = QGroupBox("Both Teams To Score (BTTS)")
        btts_layout = QHBoxLayout(self.btts_group)
        
        self.btts_yes_btn = QPushButton("Yes")
        self.btts_no_btn = QPushButton("No")
        
        btts_layout.addWidget(self.btts_yes_btn)
        btts_layout.addWidget(self.btts_no_btn)
        
        main_layout.addWidget(self.btts_group)
        
        # BTTS & Over/Under Combination
        self.btts_ou_group = QGroupBox("BTTS & Over/Under 2.5")
        btts_ou_layout = QGridLayout(self.btts_ou_group)
        
        self.btts_yes_over_btn = QPushButton("Yes & Over 2.5")
        self.btts_yes_under_btn = QPushButton("Yes & Under 2.5")
        self.btts_no_over_btn = QPushButton("No & Over 2.5")
        self.btts_no_under_btn = QPushButton("No & Under 2.5")
        
        btts_ou_layout.addWidget(self.btts_yes_over_btn, 0, 0)
        btts_ou_layout.addWidget(self.btts_yes_under_btn, 0, 1)
        btts_ou_layout.addWidget(self.btts_no_over_btn, 1, 0)
        btts_ou_layout.addWidget(self.btts_no_under_btn, 1, 1)
        
        main_layout.addWidget(self.btts_ou_group)
        
        # BTTS & 1X2 Combination
        self.btts_1x2_group = QGroupBox("BTTS & Match Result")
        btts_1x2_layout = QGridLayout(self.btts_1x2_group)
        
        self.btts_yes_home_btn = QPushButton("Yes & Home Win")
        self.btts_yes_draw_btn = QPushButton("Yes & Draw")
        self.btts_yes_away_btn = QPushButton("Yes & Away Win")
        self.btts_no_home_btn = QPushButton("No & Home Win")
        self.btts_no_draw_btn = QPushButton("No & Draw")
        self.btts_no_away_btn = QPushButton("No & Away Win")
        
        btts_1x2_layout.addWidget(self.btts_yes_home_btn, 0, 0)
        btts_1x2_layout.addWidget(self.btts_yes_draw_btn, 0, 1)
        btts_1x2_layout.addWidget(self.btts_yes_away_btn, 0, 2)
        btts_1x2_layout.addWidget(self.btts_no_home_btn, 1, 0)
        btts_1x2_layout.addWidget(self.btts_no_draw_btn, 1, 1)
        btts_1x2_layout.addWidget(self.btts_no_away_btn, 1, 2)
        
        main_layout.addWidget(self.btts_1x2_group)
        
    def update_odds(self, model, home_mean, away_mean, rho):
        """
        Update all betting market odds based on the model predictions
        
        Parameters:
        -----------
        model : DixonColesModel
            The prediction model
        home_mean, away_mean : float
            Expected goals for home and away teams
        rho : float
            Correlation parameter
        """
        # Calculate probabilities for 1X2 market
        outcomes = model.predict_match_outcome(home_mean, away_mean, rho)
        
        # Set 1X2 odds (using fair odds conversion)
        self.set_btn_odds(self.home_win_btn, outcomes['home_win'], "Home (1)")
        self.set_btn_odds(self.draw_btn, outcomes['draw'], "Draw (X)")
        self.set_btn_odds(self.away_win_btn, outcomes['away_win'], "Away (2)")
        
        # Calculate and set odds for over/under markets
        for threshold in self.ou_total_btns.keys():
            ou_probs = model.predict_over_under(home_mean, away_mean, threshold, rho)
            self.set_btn_odds(self.ou_total_btns[threshold][0], ou_probs['over'], f"Over {threshold}")
            self.set_btn_odds(self.ou_total_btns[threshold][1], ou_probs['under'], f"Under {threshold}")
            
            # Home team over/under
            home_ou_probs = model.predict_team_over_under(home_mean, threshold)
            self.set_btn_odds(self.ou_home_btns[threshold][0], home_ou_probs['over'], f"Over {threshold}")
            self.set_btn_odds(self.ou_home_btns[threshold][1], home_ou_probs['under'], f"Under {threshold}")
            
            # Away team over/under
            away_ou_probs = model.predict_team_over_under(away_mean, threshold)
            self.set_btn_odds(self.ou_away_btns[threshold][0], away_ou_probs['over'], f"Over {threshold}")
            self.set_btn_odds(self.ou_away_btns[threshold][1], away_ou_probs['under'], f"Under {threshold}")
        
        # Calculate and set BTTS odds
        btts_probs = model.predict_btts(home_mean, away_mean, rho)
        self.set_btn_odds(self.btts_yes_btn, btts_probs['yes'], "Yes")
        self.set_btn_odds(self.btts_no_btn, btts_probs['no'], "No")
        
        # Calculate and set BTTS & Over/Under odds
        btts_ou_probs = model.predict_btts_and_over_under(home_mean, away_mean, 2.5, rho)
        self.set_btn_odds(self.btts_yes_over_btn, btts_ou_probs['yes_over'], "Yes & Over 2.5")
        self.set_btn_odds(self.btts_yes_under_btn, btts_ou_probs['yes_under'], "Yes & Under 2.5")
        self.set_btn_odds(self.btts_no_over_btn, btts_ou_probs['no_over'], "No & Over 2.5")
        self.set_btn_odds(self.btts_no_under_btn, btts_ou_probs['no_under'], "No & Under 2.5")
        
        # Calculate and set BTTS & 1X2 odds
        btts_1x2_probs = model.predict_btts_and_match_result(home_mean, away_mean, rho)
        self.set_btn_odds(self.btts_yes_home_btn, btts_1x2_probs['yes_home'], "Yes & Home Win")
        self.set_btn_odds(self.btts_yes_draw_btn, btts_1x2_probs['yes_draw'], "Yes & Draw")
        self.set_btn_odds(self.btts_yes_away_btn, btts_1x2_probs['yes_away'], "Yes & Away Win")
        self.set_btn_odds(self.btts_no_home_btn, btts_1x2_probs['no_home'], "No & Home Win")
        self.set_btn_odds(self.btts_no_draw_btn, btts_1x2_probs['no_draw'], "No & Draw")
        self.set_btn_odds(self.btts_no_away_btn, btts_1x2_probs['no_away'], "No & Away Win")
    
    def set_btn_odds(self, btn, probability, label_text):
        """
        Set button text with label and odds value
        
        Parameters:
        -----------
        btn : QPushButton
            Button to update
        probability : float
            Event probability
        label_text : str
            Button label text
        """
        # Calculate "fair" odds (no margin)
        if probability > 0:
            odds = round(1.0 / probability, 2)
        else:
            odds = 999.99
        
        # Set button text with label and odds
        btn.setText(f"{label_text}\n{odds:.2f}")


class PlayerScoringWidget(QWidget):
    """
    Widget for displaying player scoring probabilities
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        
    def setup_ui(self):
        """Set up the player scoring UI"""
        main_layout = QVBoxLayout(self)
        
        # Create player input section
        player_input_layout = QHBoxLayout()
        
        # Home player input
        home_player_layout = QVBoxLayout()
        home_player_label = QLabel("Home Player:")
        self.home_player_name = QLineEdit()
        self.home_player_name.setPlaceholderText("Enter player name")
        self.home_player_name.textChanged.connect(self.update_player_names)
        
        # Add radio button group for home player input type
        home_input_type_layout = QHBoxLayout()
        self.home_input_group = QButtonGroup(self)
        self.home_input_xg = QRadioButton("Expected Goals")
        self.home_input_odds = QRadioButton("Anytime Odds")
        self.home_input_xg.setChecked(True)  # Default to expected goals
        self.home_input_group.addButton(self.home_input_xg)
        self.home_input_group.addButton(self.home_input_odds)
        home_input_type_layout.addWidget(self.home_input_xg)
        home_input_type_layout.addWidget(self.home_input_odds)
        # Connect radio buttons to update labels
        self.home_input_xg.toggled.connect(self.update_home_input_label)
        
        home_player_mean_layout = QHBoxLayout()
        self.home_player_mean_label = QLabel("Expected Goals:")
        self.home_player_mean = QDoubleSpinBox()
        self.home_player_mean.setRange(0, 5)
        self.home_player_mean.setSingleStep(0.01)
        self.home_player_mean.setDecimals(5)
        self.home_player_mean.setValue(0.5)
        home_player_mean_layout.addWidget(self.home_player_mean_label)
        home_player_mean_layout.addWidget(self.home_player_mean)
        
        # Add goal count selector
        home_goal_count_layout = QHBoxLayout()
        home_goal_count_label = QLabel("Goal Count:")
        self.home_goal_count = QComboBox()
        self.home_goal_count.addItems(["1+ Goals", "2+ Goals", "3+ Goals"])
        home_goal_count_layout.addWidget(home_goal_count_label)
        home_goal_count_layout.addWidget(self.home_goal_count)
        
        home_player_layout.addWidget(home_player_label)
        home_player_layout.addWidget(self.home_player_name)
        home_player_layout.addLayout(home_input_type_layout)
        home_player_layout.addLayout(home_player_mean_layout)
        home_player_layout.addLayout(home_goal_count_layout)
        
        # Home player expanded view button
        home_heatmap_btn = QPushButton("View Player Heatmap")
        home_heatmap_btn.clicked.connect(lambda: self.show_expanded_player_heatmap(True))
        home_player_layout.addWidget(home_heatmap_btn)
        
        # Away player input
        away_player_layout = QVBoxLayout()
        away_player_label = QLabel("Away Player:")
        self.away_player_name = QLineEdit()
        self.away_player_name.setPlaceholderText("Enter player name")
        self.away_player_name.textChanged.connect(self.update_player_names)
        
        # Add radio button group for away player input type
        away_input_type_layout = QHBoxLayout()
        self.away_input_group = QButtonGroup(self)
        self.away_input_xg = QRadioButton("Expected Goals")
        self.away_input_odds = QRadioButton("Anytime Odds")
        self.away_input_xg.setChecked(True)  # Default to expected goals
        self.away_input_group.addButton(self.away_input_xg)
        self.away_input_group.addButton(self.away_input_odds)
        away_input_type_layout.addWidget(self.away_input_xg)
        away_input_type_layout.addWidget(self.away_input_odds)
        # Connect radio buttons to update labels
        self.away_input_xg.toggled.connect(self.update_away_input_label)
        
        away_player_mean_layout = QHBoxLayout()
        self.away_player_mean_label = QLabel("Expected Goals:")
        self.away_player_mean = QDoubleSpinBox()
        self.away_player_mean.setRange(0, 5)
        self.away_player_mean.setSingleStep(0.01)
        self.away_player_mean.setDecimals(5)
        self.away_player_mean.setValue(0.5)
        away_player_mean_layout.addWidget(self.away_player_mean_label)
        away_player_mean_layout.addWidget(self.away_player_mean)
        
        # Add goal count selector
        away_goal_count_layout = QHBoxLayout()
        away_goal_count_label = QLabel("Goal Count:")
        self.away_goal_count = QComboBox()
        self.away_goal_count.addItems(["1+ Goals", "2+ Goals", "3+ Goals"])
        away_goal_count_layout.addWidget(away_goal_count_label)
        away_goal_count_layout.addWidget(self.away_goal_count)
        
        away_player_layout.addWidget(away_player_label)
        away_player_layout.addWidget(self.away_player_name)
        away_player_layout.addLayout(away_input_type_layout)
        away_player_layout.addLayout(away_player_mean_layout)
        away_player_layout.addLayout(away_goal_count_layout)
        
        # Away player expanded view button
        away_heatmap_btn = QPushButton("View Player Heatmap")
        away_heatmap_btn.clicked.connect(lambda: self.show_expanded_player_heatmap(False))
        away_player_layout.addWidget(away_heatmap_btn)
        
        player_input_layout.addLayout(home_player_layout)
        player_input_layout.addLayout(away_player_layout)
        
        # Create tab widgets for player scoring probabilities
        home_tabs = QTabWidget()
        away_tabs = QTabWidget()
        
        # HOME PLAYER TABLES
        # ==================
        
        # Basic markets tab
        home_basic_tab = QWidget()
        home_basic_layout = QVBoxLayout(home_basic_tab)
        
        # Basic markets table
        self.home_player_table = QTableWidget(10, 3)
        self.home_player_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.home_player_table.setVerticalHeaderLabels([
            "Anytime Scorer", "2+ Goals", "3+ Goals", "Scores & Team Wins", "Scores & Team Draws", 
            "Scores & Team Loses", "Scores & 1X", "Scores & X2", "Scores & 12", "2+ Goals & Team Wins"
        ])
        # Set column widths to accommodate larger buttons
        self.home_player_table.setColumnWidth(0, 200)  # Market column wider for text
        self.home_player_table.setColumnWidth(1, 60)   # Odds column
        self.home_player_table.setColumnWidth(2, 70)   # Probability column
        
        self.home_player_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.home_player_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.home_player_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better fit
        for i in range(10):
            self.home_player_table.setRowHeight(i, 50)
            
        home_basic_layout.addWidget(self.home_player_table)
        
        home_tabs.addTab(home_basic_tab, "Basic Markets")
        
        # BTTS tab
        home_btts_tab = QWidget()
        home_btts_layout = QVBoxLayout(home_btts_tab)
        
        # BTTS & Over/Under table
        self.home_btts_ou_table = QTableWidget(4, 3)
        self.home_btts_ou_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.home_btts_ou_table.setVerticalHeaderLabels([
            "Scores & BTTS & Over 2.5", "Scores & BTTS & Under 2.5", 
            "Scores & No BTTS & Over 2.5", "Scores & No BTTS & Under 2.5"
        ])
        self.home_btts_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Set column widths to accommodate larger buttons
        self.home_btts_ou_table.setColumnWidth(0, 200)  # Market column wider for text
        self.home_btts_ou_table.setColumnWidth(1, 60)   # Odds column
        self.home_btts_ou_table.setColumnWidth(2, 70)   # Probability column
        
        self.home_btts_ou_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.home_btts_ou_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.home_btts_ou_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(4):
            self.home_btts_ou_table.setRowHeight(i, 60)
            
        home_btts_layout.addWidget(self.home_btts_ou_table)
        
        # BTTS & Match Result table
        self.home_btts_result_table = QTableWidget(6, 3)
        self.home_btts_result_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.home_btts_result_table.setVerticalHeaderLabels([
            "Scores & BTTS & Home Win", "Scores & BTTS & Draw", "Scores & BTTS & Away Win",
            "Scores & No BTTS & Home Win", "Scores & No BTTS & Draw", "Scores & No BTTS & Away Win"
        ])
        self.home_btts_result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Set column widths to accommodate larger buttons
        self.home_btts_result_table.setColumnWidth(0, 200)  # Market column wider for text
        self.home_btts_result_table.setColumnWidth(1, 60)   # Odds column
        self.home_btts_result_table.setColumnWidth(2, 70)   # Probability column
        
        self.home_btts_result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.home_btts_result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.home_btts_result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(6):
            self.home_btts_result_table.setRowHeight(i, 60)
            
        home_btts_layout.addWidget(self.home_btts_result_table)
        
        home_tabs.addTab(home_btts_tab, "BTTS Combinations")
        
        # Match Over/Under tab
        home_match_ou_tab = QWidget()
        home_match_ou_layout = QVBoxLayout(home_match_ou_tab)
        
        # Player & Match Over/Under table
        self.home_match_ou_table = QTableWidget(14, 3)
        self.home_match_ou_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.home_match_ou_table.setVerticalHeaderLabels([
            "Scores & Over 0.5 Goals", "Scores & Under 0.5 Goals",
            "Scores & Over 1.5 Goals", "Scores & Under 1.5 Goals",
            "Scores & Over 2.5 Goals", "Scores & Under 2.5 Goals",
            "Scores & Over 3.5 Goals", "Scores & Under 3.5 Goals",
            "Scores & Over 4.5 Goals", "Scores & Under 4.5 Goals",
            "Scores & Over 5.5 Goals", "Scores & Under 5.5 Goals",
            "Scores & Over 6.5 Goals", "Scores & Under 6.5 Goals"
        ])
        self.home_match_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Set column widths to accommodate larger buttons
        self.home_match_ou_table.setColumnWidth(0, 200)  # Market column wider for text
        self.home_match_ou_table.setColumnWidth(1, 60)   # Odds column
        self.home_match_ou_table.setColumnWidth(2, 70)   # Probability column
        
        self.home_match_ou_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.home_match_ou_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.home_match_ou_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(14):
            self.home_match_ou_table.setRowHeight(i, 60)
        
        home_match_ou_layout.addWidget(self.home_match_ou_table)
        
        home_tabs.addTab(home_match_ou_tab, "Match Over/Under")
        
        # Home Team Over/Under tab
        home_team_ou_tab = QWidget()
        home_team_ou_layout = QVBoxLayout(home_team_ou_tab)
        
        # Player & Home Team Over/Under table
        self.home_team_ou_table = QTableWidget(14, 3)
        self.home_team_ou_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.home_team_ou_table.setVerticalHeaderLabels([
            "Scores & Home Over 0.5", "Scores & Home Under 0.5",
            "Scores & Home Over 1.5", "Scores & Home Under 1.5",
            "Scores & Home Over 2.5", "Scores & Home Under 2.5",
            "Scores & Home Over 3.5", "Scores & Home Under 3.5",
            "Scores & Home Over 4.5", "Scores & Home Under 4.5",
            "Scores & Home Over 5.5", "Scores & Home Under 5.5",
            "Scores & Home Over 6.5", "Scores & Home Under 6.5"
        ])
        self.home_team_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Set column widths to accommodate larger buttons
        self.home_team_ou_table.setColumnWidth(0, 200)  # Market column wider for text
        self.home_team_ou_table.setColumnWidth(1, 60)   # Odds column
        self.home_team_ou_table.setColumnWidth(2, 70)   # Probability column
        
        self.home_team_ou_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.home_team_ou_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.home_team_ou_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(14):
            self.home_team_ou_table.setRowHeight(i, 60)
            
        home_team_ou_layout.addWidget(self.home_team_ou_table)
        
        home_tabs.addTab(home_team_ou_tab, "Home Team O/U")
        
        # Away Team Over/Under tab
        home_away_ou_tab = QWidget()
        home_away_ou_layout = QVBoxLayout(home_away_ou_tab)
        
        # Player & Away Team Over/Under table
        self.home_away_ou_table = QTableWidget(14, 3)
        self.home_away_ou_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.home_away_ou_table.setVerticalHeaderLabels([
            "Scores & Away Over 0.5", "Scores & Away Under 0.5",
            "Scores & Away Over 1.5", "Scores & Away Under 1.5",
            "Scores & Away Over 2.5", "Scores & Away Under 2.5",
            "Scores & Away Over 3.5", "Scores & Away Under 3.5",
            "Scores & Away Over 4.5", "Scores & Away Under 4.5",
            "Scores & Away Over 5.5", "Scores & Away Under 5.5",
            "Scores & Away Over 6.5", "Scores & Away Under 6.5"
        ])
        self.home_away_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Set column widths to accommodate larger buttons
        self.home_away_ou_table.setColumnWidth(0, 200)  # Market column wider for text
        self.home_away_ou_table.setColumnWidth(1, 60)   # Odds column
        self.home_away_ou_table.setColumnWidth(2, 70)   # Probability column
        
        self.home_away_ou_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.home_away_ou_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.home_away_ou_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(14):
            self.home_away_ou_table.setRowHeight(i, 60)
            
        home_away_ou_layout.addWidget(self.home_away_ou_table)
        
        home_tabs.addTab(home_away_ou_tab, "Away Team O/U")
        
        # Player Over/Under tab
        home_player_ou_tab = QWidget()
        home_player_ou_layout = QVBoxLayout(home_player_ou_tab)
        
        # Over/Under table
        self.home_ou_table = QTableWidget(7, 5)
        self.home_ou_table.setHorizontalHeaderLabels(["Market", "Over Odds", "Over %", "Under Odds", "Under %"])
        self.home_ou_table.setVerticalHeaderLabels([
            "Over/Under 0.5", "Over/Under 1.5", "Over/Under 2.5", "Over/Under 3.5", 
            "Over/Under 4.5", "Over/Under 5.5", "Over/Under 6.5"
        ])
        self.home_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        home_player_ou_layout.addWidget(self.home_ou_table)
        
        home_tabs.addTab(home_player_ou_tab, "Player O/U")
        
        # Exact goals tab
        home_exact_tab = QWidget()
        home_exact_layout = QVBoxLayout(home_exact_tab)
        
        # Exact goals table
        self.home_exact_table = QTableWidget(6, 3)
        self.home_exact_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.home_exact_table.setVerticalHeaderLabels([
            "0 Goals", "1 Goal", "2 Goals", "3 Goals", "4 Goals", "5 Goals"
        ])
        self.home_exact_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        home_exact_layout.addWidget(self.home_exact_table)
        
        home_tabs.addTab(home_exact_tab, "Exact Goals")
        
        # Player Scoring Heatmap tab
        home_heatmap_tab = QWidget()
        home_heatmap_layout = QVBoxLayout(home_heatmap_tab)
        
        # Add description
        home_heatmap_description = QLabel("Player Scoring Probability Heatmap")
        home_heatmap_description.setAlignment(Qt.AlignCenter)
        home_heatmap_description.setStyleSheet("font-size: 14px; font-weight: bold;")
        home_heatmap_layout.addWidget(home_heatmap_description)
        
        # Create MatplotlibCanvas for the heatmap
        self.home_heatmap_canvas = MatplotlibCanvas(width=8, height=6)
        home_heatmap_layout.addWidget(self.home_heatmap_canvas)
        
        home_tabs.addTab(home_heatmap_tab, "Scoring Heatmap")
        
        # AWAY PLAYER TABLES
        # ==================
        
        # Basic markets tab
        away_basic_tab = QWidget()
        away_basic_layout = QVBoxLayout(away_basic_tab)
        
        # Basic markets table
        self.away_player_table = QTableWidget(10, 3)
        self.away_player_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.away_player_table.setVerticalHeaderLabels([
            "Anytime Scorer", "2+ Goals", "3+ Goals", "Scores & Team Wins", "Scores & Team Draws", 
            "Scores & Team Loses", "Scores & 1X", "Scores & X2", "Scores & 12", "2+ Goals & Team Wins"
        ])
        # Set column widths to accommodate larger buttons
        self.away_player_table.setColumnWidth(0, 200)  # Market column wider for text
        self.away_player_table.setColumnWidth(1, 60)   # Odds column
        self.away_player_table.setColumnWidth(2, 70)   # Probability column
        
        self.away_player_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.away_player_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.away_player_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better fit
        for i in range(10):
            self.away_player_table.setRowHeight(i, 50)
            
        away_basic_layout.addWidget(self.away_player_table)
        
        away_tabs.addTab(away_basic_tab, "Basic Markets")
        
        # BTTS tab
        away_btts_tab = QWidget()
        away_btts_layout = QVBoxLayout(away_btts_tab)
        
        # BTTS & Over/Under table
        self.away_btts_ou_table = QTableWidget(4, 3)
        self.away_btts_ou_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.away_btts_ou_table.setVerticalHeaderLabels([
            "Scores & BTTS & Over 2.5", "Scores & BTTS & Under 2.5", 
            "Scores & No BTTS & Over 2.5", "Scores & No BTTS & Under 2.5"
        ])
        self.away_btts_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Set column widths to accommodate larger buttons
        self.away_btts_ou_table.setColumnWidth(0, 200)  # Market column wider for text
        self.away_btts_ou_table.setColumnWidth(1, 60)   # Odds column
        self.away_btts_ou_table.setColumnWidth(2, 70)   # Probability column
        
        self.away_btts_ou_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.away_btts_ou_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.away_btts_ou_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(4):
            self.away_btts_ou_table.setRowHeight(i, 60)
            
        away_btts_layout.addWidget(self.away_btts_ou_table)
        
        # BTTS & Match Result table
        self.away_btts_result_table = QTableWidget(6, 3)
        self.away_btts_result_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.away_btts_result_table.setVerticalHeaderLabels([
            "Scores & BTTS & Home Win", "Scores & BTTS & Draw", "Scores & BTTS & Away Win",
            "Scores & No BTTS & Home Win", "Scores & No BTTS & Draw", "Scores & No BTTS & Away Win"
        ])
        self.away_btts_result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Set column widths to accommodate larger buttons
        self.away_btts_result_table.setColumnWidth(0, 200)  # Market column wider for text
        self.away_btts_result_table.setColumnWidth(1, 60)   # Odds column
        self.away_btts_result_table.setColumnWidth(2, 70)   # Probability column
        
        self.away_btts_result_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.away_btts_result_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.away_btts_result_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(6):
            self.away_btts_result_table.setRowHeight(i, 60)
            
        away_btts_layout.addWidget(self.away_btts_result_table)
        
        away_tabs.addTab(away_btts_tab, "BTTS Combinations")
        
        # Match Over/Under tab
        away_match_ou_tab = QWidget()
        away_match_ou_layout = QVBoxLayout(away_match_ou_tab)
        
        # Player & Match Over/Under table
        self.away_match_ou_table = QTableWidget(14, 3)
        self.away_match_ou_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.away_match_ou_table.setVerticalHeaderLabels([
            "Scores & Over 0.5 Goals", "Scores & Under 0.5 Goals",
            "Scores & Over 1.5 Goals", "Scores & Under 1.5 Goals",
            "Scores & Over 2.5 Goals", "Scores & Under 2.5 Goals",
            "Scores & Over 3.5 Goals", "Scores & Under 3.5 Goals",
            "Scores & Over 4.5 Goals", "Scores & Under 4.5 Goals",
            "Scores & Over 5.5 Goals", "Scores & Under 5.5 Goals",
            "Scores & Over 6.5 Goals", "Scores & Under 6.5 Goals"
        ])
        self.away_match_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Set column widths to accommodate larger buttons
        self.away_match_ou_table.setColumnWidth(0, 200)  # Market column wider for text
        self.away_match_ou_table.setColumnWidth(1, 60)   # Odds column
        self.away_match_ou_table.setColumnWidth(2, 70)   # Probability column
        
        self.away_match_ou_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.away_match_ou_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.away_match_ou_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(14):
            self.away_match_ou_table.setRowHeight(i, 60)
        
        away_match_ou_layout.addWidget(self.away_match_ou_table)
        
        away_tabs.addTab(away_match_ou_tab, "Match Over/Under")
        
        # Home Team Over/Under tab
        away_home_ou_tab = QWidget()
        away_home_ou_layout = QVBoxLayout(away_home_ou_tab)
        
        # Player & Home Team Over/Under table
        self.away_home_ou_table = QTableWidget(14, 3)
        self.away_home_ou_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.away_home_ou_table.setVerticalHeaderLabels([
            "Scores & Home Over 0.5", "Scores & Home Under 0.5",
            "Scores & Home Over 1.5", "Scores & Home Under 1.5",
            "Scores & Home Over 2.5", "Scores & Home Under 2.5",
            "Scores & Home Over 3.5", "Scores & Home Under 3.5",
            "Scores & Home Over 4.5", "Scores & Home Under 4.5",
            "Scores & Home Over 5.5", "Scores & Home Under 5.5",
            "Scores & Home Over 6.5", "Scores & Home Under 6.5"
        ])
        self.away_home_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Set column widths to accommodate larger buttons
        self.away_home_ou_table.setColumnWidth(0, 200)  # Market column wider for text
        self.away_home_ou_table.setColumnWidth(1, 60)   # Odds column
        self.away_home_ou_table.setColumnWidth(2, 70)   # Probability column
        
        self.away_home_ou_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.away_home_ou_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.away_home_ou_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(14):
            self.away_home_ou_table.setRowHeight(i, 60)
            
        away_home_ou_layout.addWidget(self.away_home_ou_table)
        
        away_tabs.addTab(away_home_ou_tab, "Home Team O/U")
        
        # Away Team Over/Under tab
        away_team_ou_tab = QWidget()
        away_team_ou_layout = QVBoxLayout(away_team_ou_tab)
        
        # Player & Away Team Over/Under table
        self.away_team_ou_table = QTableWidget(14, 3)
        self.away_team_ou_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.away_team_ou_table.setVerticalHeaderLabels([
            "Scores & Away Over 0.5", "Scores & Away Under 0.5",
            "Scores & Away Over 1.5", "Scores & Away Under 1.5",
            "Scores & Away Over 2.5", "Scores & Away Under 2.5",
            "Scores & Away Over 3.5", "Scores & Away Under 3.5",
            "Scores & Away Over 4.5", "Scores & Away Under 4.5",
            "Scores & Away Over 5.5", "Scores & Away Under 5.5",
            "Scores & Away Over 6.5", "Scores & Away Under 6.5"
        ])
        self.away_team_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        # Set column widths to accommodate larger buttons
        self.away_team_ou_table.setColumnWidth(0, 200)  # Market column wider for text
        self.away_team_ou_table.setColumnWidth(1, 60)   # Odds column
        self.away_team_ou_table.setColumnWidth(2, 70)   # Probability column
        
        self.away_team_ou_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Fixed)  # Market column fixed width
        self.away_team_ou_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)  # Odds column stretches
        self.away_team_ou_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)  # Probability column stretches
        
        # Set row heights for better visibility
        for i in range(14):
            self.away_team_ou_table.setRowHeight(i, 60)
            
        away_team_ou_layout.addWidget(self.away_team_ou_table)
        
        away_tabs.addTab(away_team_ou_tab, "Away Team O/U")
        
        # Player Over/Under tab
        away_player_ou_tab = QWidget()
        away_player_ou_layout = QVBoxLayout(away_player_ou_tab)
        
        # Over/Under table
        self.away_ou_table = QTableWidget(7, 5)
        self.away_ou_table.setHorizontalHeaderLabels(["Market", "Over Odds", "Over %", "Under Odds", "Under %"])
        self.away_ou_table.setVerticalHeaderLabels([
            "Over/Under 0.5", "Over/Under 1.5", "Over/Under 2.5", "Over/Under 3.5", 
            "Over/Under 4.5", "Over/Under 5.5", "Over/Under 6.5"
        ])
        self.away_ou_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        away_player_ou_layout.addWidget(self.away_ou_table)
        
        away_tabs.addTab(away_player_ou_tab, "Player O/U")
        
        # Exact goals tab
        away_exact_tab = QWidget()
        away_exact_layout = QVBoxLayout(away_exact_tab)
        
        # Exact goals table
        self.away_exact_table = QTableWidget(6, 3)
        self.away_exact_table.setHorizontalHeaderLabels(["Market", "Odds", "Prob"])
        self.away_exact_table.setVerticalHeaderLabels([
            "0 Goals", "1 Goal", "2 Goals", "3 Goals", "4 Goals", "5 Goals"
        ])
        self.away_exact_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        away_exact_layout.addWidget(self.away_exact_table)
        
        away_tabs.addTab(away_exact_tab, "Exact Goals")
        
        # Player Scoring Heatmap tab
        away_heatmap_tab = QWidget()
        away_heatmap_layout = QVBoxLayout(away_heatmap_tab)
        
        # Add description
        away_heatmap_description = QLabel("Player Scoring Probability Heatmap")
        away_heatmap_description.setAlignment(Qt.AlignCenter)
        away_heatmap_description.setStyleSheet("font-size: 14px; font-weight: bold;")
        away_heatmap_layout.addWidget(away_heatmap_description)
        
        # Create MatplotlibCanvas for the heatmap
        self.away_heatmap_canvas = MatplotlibCanvas(width=8, height=6)
        away_heatmap_layout.addWidget(self.away_heatmap_canvas)
        
        away_tabs.addTab(away_heatmap_tab, "Scoring Heatmap")
        
        # Add home and away tabs to layout
        home_group = QGroupBox("Home Player")
        home_group_layout = QVBoxLayout(home_group)
        home_group_layout.addWidget(home_tabs)
        
        away_group = QGroupBox("Away Player")
        away_group_layout = QVBoxLayout(away_group)
        away_group_layout.addWidget(away_tabs)
        
        tables_layout = QHBoxLayout()
        tables_layout.addWidget(home_group)
        tables_layout.addWidget(away_group)
        
        # Add all layouts to the main layout
        main_layout.addLayout(player_input_layout)
        main_layout.addLayout(tables_layout)
        
        self.setLayout(main_layout)
    
    def update_home_input_label(self):
        """Update home player input label based on selected input type"""
        if self.home_input_xg.isChecked():
            self.home_player_mean_label.setText("Expected Goals:")
            self.home_player_mean.setDecimals(5)
            self.home_player_mean.setRange(0, 5)
            self.home_player_mean.setSingleStep(0.01)
        else:
            self.home_player_mean_label.setText("Anytime Odds:")
            self.home_player_mean.setDecimals(2)
            self.home_player_mean.setRange(1.01, 100)
            self.home_player_mean.setSingleStep(0.1)
            self.home_player_mean.setValue(3.00)  # Default odds value
            
    def update_away_input_label(self):
        """Update away player input label based on selected input type"""
        if self.away_input_xg.isChecked():
            self.away_player_mean_label.setText("Expected Goals:")
            self.away_player_mean.setDecimals(5)
            self.away_player_mean.setRange(0, 5)
            self.away_player_mean.setSingleStep(0.01)
        else:
            self.away_player_mean_label.setText("Anytime Odds:")
            self.away_player_mean.setDecimals(2)
            self.away_player_mean.setRange(1.01, 100)
            self.away_player_mean.setSingleStep(0.1)
            self.away_player_mean.setValue(3.00)  # Default odds value
            
    def get_home_player_xg(self, home_mean):
        """Get home player xG based on input type"""
        if self.home_input_xg.isChecked():
            # Direct xG input
            return self.home_player_mean.value()
        else:
            # Calculate xG from odds
            odds_value = self.home_player_mean.value()
            return calculate_player_mean_from_odds(odds_value, home_mean)
            
    def get_away_player_xg(self, away_mean):
        """Get away player xG based on input type"""
        if self.away_input_xg.isChecked():
            # Direct xG input
            return self.away_player_mean.value()
        else:
            # Calculate xG from odds
            odds_value = self.away_player_mean.value()
            return calculate_player_mean_from_odds(odds_value, away_mean)
            
    def update_player_odds(self, model, home_mean, away_mean, rho, score_grid=None):
        """Update player scoring odds based on model predictions"""
        # Get player means (directly or calculated from odds)
        home_player_mean = self.get_home_player_xg(home_mean)
        away_player_mean = self.get_away_player_xg(away_mean)
        
        # Get team names from parent MainWindow
        main_window = self.window()
        home_team_name = main_window.home_team_name.text()
        away_team_name = main_window.away_team_name.text()
        
        # Get goal count selections
        home_goal_count_index = self.home_goal_count.currentIndex()
        away_goal_count_index = self.away_goal_count.currentIndex()
        
        # Always calculate all basic player scoring probabilities
        home_scorer_1plus = model.predict_player_scorer(home_player_mean, home_mean, score_grid=score_grid, is_home_team=True)
        home_2plus_prob = model.predict_player_2plus(home_player_mean, home_mean, score_grid=score_grid, is_home_team=True)
        home_3plus_prob = model.predict_player_3plus(home_player_mean, home_mean, score_grid=score_grid, is_home_team=True)
        
        away_scorer_1plus = model.predict_player_scorer(away_player_mean, away_mean, score_grid=score_grid, is_home_team=False)
        away_2plus_prob = model.predict_player_2plus(away_player_mean, away_mean, score_grid=score_grid, is_home_team=False)
        away_3plus_prob = model.predict_player_3plus(away_player_mean, away_mean, score_grid=score_grid, is_home_team=False)
        
        # For combination markets, use the probability based on goal count selection
        if home_goal_count_index == 0:  # 1+ goals
            home_scorer_prob = home_scorer_1plus
        elif home_goal_count_index == 1:  # 2+ goals
            home_scorer_prob = home_2plus_prob
        else:  # 3+ goals
            home_scorer_prob = home_3plus_prob
            
        if away_goal_count_index == 0:  # 1+ goals
            away_scorer_prob = away_scorer_1plus
        elif away_goal_count_index == 1:  # 2+ goals
            away_scorer_prob = away_2plus_prob
        else:  # 3+ goals
            away_scorer_prob = away_3plus_prob
        
        # Calculate scorer and team result probabilities
        # For these and all combinations below, we'll use the selected goal count probability
        if home_goal_count_index == 0:  # 1+ goals
            home_scorer_result = model.predict_player_scorer_and_result(
                home_player_mean, home_mean, home_mean, away_mean, True, rho, score_grid=score_grid
            )
        elif home_goal_count_index == 1:  # 2+ goals
            # Calculate directly with the new method for 2+ goals
            home_scorer_result = model.predict_player_scorer_with_nplus_and_result(
                home_player_mean, home_mean, home_mean, away_mean, True, rho, 2, score_grid=score_grid
            )
        else:  # 3+ goals
            # Calculate directly with the new method for 3+ goals
            home_scorer_result = model.predict_player_scorer_with_nplus_and_result(
                home_player_mean, home_mean, home_mean, away_mean, True, rho, 3, score_grid=score_grid
            )
            
        if away_goal_count_index == 0:  # 1+ goals
            away_scorer_result = model.predict_player_scorer_and_result(
                away_player_mean, away_mean, home_mean, away_mean, False, rho, score_grid=score_grid
            )
        elif away_goal_count_index == 1:  # 2+ goals
            # Calculate directly with the new method for 2+ goals
            away_scorer_result = model.predict_player_scorer_with_nplus_and_result(
                away_player_mean, away_mean, home_mean, away_mean, False, rho, 2, score_grid=score_grid
            )
        else:  # 3+ goals
            # Calculate directly with the new method for 3+ goals
            away_scorer_result = model.predict_player_scorer_with_nplus_and_result(
                away_player_mean, away_mean, home_mean, away_mean, False, rho, 3, score_grid=score_grid
            )
        
        # Calculate scorer and double chance probabilities based on scorer_result
        home_scorer_dc = {
            'scores_1X': home_scorer_result["scores_team_wins"] + home_scorer_result["scores_team_draws"],
            'scores_X2': home_scorer_result["scores_team_draws"] + home_scorer_result["scores_team_loses"],
            'scores_12': home_scorer_result["scores_team_wins"] + home_scorer_result["scores_team_loses"]
        }
        
        away_scorer_dc = {
            'scores_1X': away_scorer_result["scores_team_loses"] + away_scorer_result["scores_team_draws"],
            'scores_X2': away_scorer_result["scores_team_draws"] + away_scorer_result["scores_team_wins"],
            'scores_12': away_scorer_result["scores_team_loses"] + away_scorer_result["scores_team_wins"]
        }
        
        # Populate the home player basic markets table
        self.set_table_item(self.home_player_table, 0, home_scorer_prob, "Anytime Scorer", home_goal_count_index)
        self.set_table_item(self.home_player_table, 1, home_2plus_prob, "2+ Goals")
        self.set_table_item(self.home_player_table, 2, home_3plus_prob, "3+ Goals")
        self.set_table_item(self.home_player_table, 3, home_scorer_result["scores_team_wins"], "Scores & Team Wins", home_goal_count_index)
        self.set_table_item(self.home_player_table, 4, home_scorer_result["scores_team_draws"], "Scores & Team Draws", home_goal_count_index)
        self.set_table_item(self.home_player_table, 5, home_scorer_result["scores_team_loses"], "Scores & Team Loses", home_goal_count_index)
        self.set_table_item(self.home_player_table, 6, home_scorer_dc["scores_1X"], "Scores & 1X", home_goal_count_index)
        self.set_table_item(self.home_player_table, 7, home_scorer_dc["scores_X2"], "Scores & X2", home_goal_count_index)
        self.set_table_item(self.home_player_table, 8, home_scorer_dc["scores_12"], "Scores & 12", home_goal_count_index)
        
        # For 2+ Goals & Team Wins, adjust or calculate directly
        if home_goal_count_index == 0:  # 1+ goals selected, so adjust the 2+ probability
            two_plus_team_wins = home_scorer_result["scores_team_wins"] * (home_2plus_prob / home_scorer_1plus) if home_scorer_1plus > 0 else 0
            self.set_table_item(self.home_player_table, 9, two_plus_team_wins, "2+ Goals & Team Wins")
        elif home_goal_count_index == 1:  # 2+ goals already selected
            self.set_table_item(self.home_player_table, 9, home_scorer_result["scores_team_wins"], "2+ Goals & Team Wins")
        else:  # 3+ goals selected, this is a subset of 2+ goals
            two_plus_result = model.predict_player_scorer_with_nplus_and_result(
                home_player_mean, home_mean, home_mean, away_mean, True, rho, 2, score_grid=score_grid
            )
            self.set_table_item(self.home_player_table, 9, two_plus_result["scores_team_wins"], "2+ Goals & Team Wins")
            
        # Populate the away player basic markets table
        self.set_table_item(self.away_player_table, 0, away_scorer_prob, "Anytime Scorer", away_goal_count_index)
        self.set_table_item(self.away_player_table, 1, away_2plus_prob, "2+ Goals")
        self.set_table_item(self.away_player_table, 2, away_3plus_prob, "3+ Goals")
        self.set_table_item(self.away_player_table, 3, away_scorer_result["scores_team_wins"], "Scores & Team Wins", away_goal_count_index)
        self.set_table_item(self.away_player_table, 4, away_scorer_result["scores_team_draws"], "Scores & Team Draws", away_goal_count_index)
        self.set_table_item(self.away_player_table, 5, away_scorer_result["scores_team_loses"], "Scores & Team Loses", away_goal_count_index)
        self.set_table_item(self.away_player_table, 6, away_scorer_dc["scores_1X"], "Scores & 1X", away_goal_count_index)
        self.set_table_item(self.away_player_table, 7, away_scorer_dc["scores_X2"], "Scores & X2", away_goal_count_index)
        self.set_table_item(self.away_player_table, 8, away_scorer_dc["scores_12"], "Scores & 12", away_goal_count_index)
        
        # For 2+ Goals & Team Wins, adjust or calculate directly
        if away_goal_count_index == 0:  # 1+ goals selected, so adjust the 2+ probability
            two_plus_team_wins = away_scorer_result["scores_team_wins"] * (away_2plus_prob / away_scorer_1plus) if away_scorer_1plus > 0 else 0
            self.set_table_item(self.away_player_table, 9, two_plus_team_wins, "2+ Goals & Team Wins")
        elif away_goal_count_index == 1:  # 2+ goals already selected
            self.set_table_item(self.away_player_table, 9, away_scorer_result["scores_team_wins"], "2+ Goals & Team Wins")
        else:  # 3+ goals selected, this is a subset of 2+ goals
            two_plus_result = model.predict_player_scorer_with_nplus_and_result(
                away_player_mean, away_mean, home_mean, away_mean, False, rho, 2, score_grid=score_grid
            )
            self.set_table_item(self.away_player_table, 9, two_plus_result["scores_team_wins"], "2+ Goals & Team Wins")
            
        # Calculate BTTS & Over/Under probabilities
        try:
            # Home player BTTS & O/U
            if home_goal_count_index == 0:  # 1+ goals
                home_btts_ou = model.predict_player_scorer_with_btts_and_ou(
                    home_player_mean, home_mean, home_mean, away_mean, 2.5, True, rho, score_grid=score_grid
                )
            else:  # 2+ or 3+ goals
                home_btts_ou = model.predict_player_scorer_with_nplus_btts_and_ou(
                    home_player_mean, home_mean, home_mean, away_mean, 2.5, True, rho, 
                    2 if home_goal_count_index == 1 else 3, score_grid=score_grid
                )
            
            # Away player BTTS & O/U
            if away_goal_count_index == 0:  # 1+ goals
                away_btts_ou = model.predict_player_scorer_with_btts_and_ou(
                    away_player_mean, away_mean, home_mean, away_mean, 2.5, False, rho, score_grid=score_grid
                )
            else:  # 2+ or 3+ goals
                away_btts_ou = model.predict_player_scorer_with_nplus_btts_and_ou(
                    away_player_mean, away_mean, home_mean, away_mean, 2.5, False, rho,
                    2 if away_goal_count_index == 1 else 3, score_grid=score_grid
                )
            
            # BTTS & Match Result for home player
            if home_goal_count_index == 0:  # 1+ goals
                home_btts_result = model.predict_player_scorer_with_btts_and_match_result(
                    home_player_mean, home_mean, home_mean, away_mean, True, rho, score_grid=score_grid
                )
            else:  # 2+ or 3+ goals
                home_btts_result = model.predict_player_scorer_with_nplus_btts_and_match_result(
                    home_player_mean, home_mean, home_mean, away_mean, True, rho,
                    2 if home_goal_count_index == 1 else 3, score_grid=score_grid
                )
            
            # BTTS & Match Result for away player
            if away_goal_count_index == 0:  # 1+ goals
                away_btts_result = model.predict_player_scorer_with_btts_and_match_result(
                    away_player_mean, away_mean, home_mean, away_mean, False, rho, score_grid=score_grid
                )
            else:  # 2+ or 3+ goals
                away_btts_result = model.predict_player_scorer_with_nplus_btts_and_match_result(
                    away_player_mean, away_mean, home_mean, away_mean, False, rho,
                    2 if away_goal_count_index == 1 else 3, score_grid=score_grid
                )
            
            # Adjust BTTS & Match Result combinations for 2+/3+ if needed
            if home_goal_count_index > 0:
                home_btts_result = self._adjust_market_for_nplus(home_btts_result, home_scorer_1plus, home_scorer_prob)
                
            if away_goal_count_index > 0:
                away_btts_result = self._adjust_market_for_nplus(away_btts_result, away_scorer_1plus, away_scorer_prob)
            
            # Populate the BTTS & Match Result tables
            # Home player BTTS & Match Result
            self.set_table_item(self.home_btts_result_table, 0, home_btts_result["scores_yes_home"], "Scores & BTTS & Home Win", home_goal_count_index)
            self.set_table_item(self.home_btts_result_table, 1, home_btts_result["scores_yes_draw"], "Scores & BTTS & Draw", home_goal_count_index)
            self.set_table_item(self.home_btts_result_table, 2, home_btts_result["scores_yes_away"], "Scores & BTTS & Away Win", home_goal_count_index)
            self.set_table_item(self.home_btts_result_table, 3, home_btts_result["scores_no_home"], "Scores & No BTTS & Home Win", home_goal_count_index)
            self.set_table_item(self.home_btts_result_table, 4, home_btts_result["scores_no_draw"], "Scores & No BTTS & Draw", home_goal_count_index)
            self.set_table_item(self.home_btts_result_table, 5, home_btts_result["scores_no_away"], "Scores & No BTTS & Away Win", home_goal_count_index)
            
            # Away player BTTS & Match Result
            self.set_table_item(self.away_btts_result_table, 0, away_btts_result["scores_yes_home"], "Scores & BTTS & Home Win", away_goal_count_index)
            self.set_table_item(self.away_btts_result_table, 1, away_btts_result["scores_yes_draw"], "Scores & BTTS & Draw", away_goal_count_index)
            self.set_table_item(self.away_btts_result_table, 2, away_btts_result["scores_yes_away"], "Scores & BTTS & Away Win", away_goal_count_index)
            self.set_table_item(self.away_btts_result_table, 3, away_btts_result["scores_no_home"], "Scores & No BTTS & Home Win", away_goal_count_index)
            self.set_table_item(self.away_btts_result_table, 4, away_btts_result["scores_no_draw"], "Scores & No BTTS & Draw", away_goal_count_index)
            self.set_table_item(self.away_btts_result_table, 5, away_btts_result["scores_no_away"], "Scores & No BTTS & Away Win", away_goal_count_index)
            
            # Populate BTTS & Over/Under tables
            # Home player BTTS & Over/Under
            self.set_table_item(self.home_btts_ou_table, 0, home_btts_ou["yes_over"], "Scores & BTTS & Over 2.5", home_goal_count_index)
            self.set_table_item(self.home_btts_ou_table, 1, home_btts_ou["yes_under"], "Scores & BTTS & Under 2.5", home_goal_count_index)
            self.set_table_item(self.home_btts_ou_table, 2, home_btts_ou["no_over"], "Scores & No BTTS & Over 2.5", home_goal_count_index)
            self.set_table_item(self.home_btts_ou_table, 3, home_btts_ou["no_under"], "Scores & No BTTS & Under 2.5", home_goal_count_index)
            
            # Away player BTTS & Over/Under
            self.set_table_item(self.away_btts_ou_table, 0, away_btts_ou["yes_over"], "Scores & BTTS & Over 2.5", away_goal_count_index)
            self.set_table_item(self.away_btts_ou_table, 1, away_btts_ou["yes_under"], "Scores & BTTS & Under 2.5", away_goal_count_index)
            self.set_table_item(self.away_btts_ou_table, 2, away_btts_ou["no_over"], "Scores & No BTTS & Over 2.5", away_goal_count_index)
            self.set_table_item(self.away_btts_ou_table, 3, away_btts_ou["no_under"], "Scores & No BTTS & Under 2.5", away_goal_count_index)
            
            # Calculate Over/Under probabilities for different thresholds
            thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
            
            # Create dictionaries to store all over/under probabilities for home and away players
            home_ou_probabilities = {"team": home_team_name}
            away_ou_probabilities = {"team": away_team_name}
            
            # Calculate Match Over/Under probabilities
            for threshold in thresholds:
                # Home player
                home_match_ou = model.predict_player_scorer_with_match_over_under(
                    home_player_mean, home_mean, home_mean, away_mean, threshold, True, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if home_goal_count_index > 0:
                    home_match_ou = self._adjust_market_for_nplus(home_match_ou, home_scorer_1plus, home_scorer_prob)
                    
                for key, value in home_match_ou.items():
                    home_ou_probabilities[key] = value
                    
                # Away player
                away_match_ou = model.predict_player_scorer_with_match_over_under(
                    away_player_mean, away_mean, home_mean, away_mean, threshold, False, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if away_goal_count_index > 0:
                    away_match_ou = self._adjust_market_for_nplus(away_match_ou, away_scorer_1plus, away_scorer_prob)
                    
                for key, value in away_match_ou.items():
                    away_ou_probabilities[key] = value
            
            # Calculate Home Team Over/Under probabilities
            for threshold in thresholds:
                # Home player
                home_home_ou = model.predict_player_scorer_with_home_team_over_under(
                    home_player_mean, home_mean, home_mean, away_mean, threshold, True, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if home_goal_count_index > 0:
                    home_home_ou = self._adjust_market_for_nplus(home_home_ou, home_scorer_1plus, home_scorer_prob)
                    
                for key, value in home_home_ou.items():
                    home_ou_probabilities[key] = value
                    
                # Away player
                away_home_ou = model.predict_player_scorer_with_home_team_over_under(
                    away_player_mean, away_mean, home_mean, away_mean, threshold, False, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if away_goal_count_index > 0:
                    away_home_ou = self._adjust_market_for_nplus(away_home_ou, away_scorer_1plus, away_scorer_prob)
                    
                for key, value in away_home_ou.items():
                    away_ou_probabilities[key] = value
            
            # Calculate Away Team Over/Under probabilities
            for threshold in thresholds:
                # Home player
                home_away_ou = model.predict_player_scorer_with_away_team_over_under(
                    home_player_mean, home_mean, home_mean, away_mean, threshold, True, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if home_goal_count_index > 0:
                    home_away_ou = self._adjust_market_for_nplus(home_away_ou, home_scorer_1plus, home_scorer_prob)
                    
                for key, value in home_away_ou.items():
                    home_ou_probabilities[key] = value
                    
                # Away player
                away_away_ou = model.predict_player_scorer_with_away_team_over_under(
                    away_player_mean, away_mean, home_mean, away_mean, threshold, False, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if away_goal_count_index > 0:
                    away_away_ou = self._adjust_market_for_nplus(away_away_ou, away_scorer_1plus, away_scorer_prob)
                    
                for key, value in away_away_ou.items():
                    away_ou_probabilities[key] = value
            
            # Update player over/under tables
            for i, threshold in enumerate(thresholds):
                home_ou_prob = model.predict_player_over_under(home_player_mean, threshold, score_grid=score_grid, is_home_team=True)
                away_ou_prob = model.predict_player_over_under(away_player_mean, threshold, score_grid=score_grid, is_home_team=False)
                
                self.set_ou_table_item(self.home_ou_table, i, home_ou_prob, f"Over/Under {threshold}")
                self.set_ou_table_item(self.away_ou_table, i, away_ou_prob, f"Over/Under {threshold}")
            
            # Update exact goals tables
            for goals in range(6):
                # For exact goals, we're showing the true probability, no adjustments needed
                home_exact_prob = model.predict_player_exact_goals(home_player_mean, home_mean, goals, score_grid=score_grid, is_home_team=True)
                away_exact_prob = model.predict_player_exact_goals(away_player_mean, away_mean, goals, score_grid=score_grid, is_home_team=False)
                self.set_exact_goals_item(self.home_exact_table, goals, home_exact_prob)
                self.set_exact_goals_item(self.away_exact_table, goals, away_exact_prob)
            
            # Update player scoring heatmaps
            self.update_player_scoring_heatmap(model, home_player_mean, home_mean, away_mean, rho, True, home_team_name, away_team_name, score_grid)
            self.update_player_scoring_heatmap(model, away_player_mean, away_mean, home_mean, rho, False, home_team_name, away_team_name, score_grid)
            
            # Populate Match Over/Under tables
            # Home player Match Over/Under
            row = 0
            for threshold in thresholds:
                # Get probabilities for this threshold
                # Get the match over/under result for this specific threshold
                home_match_ou = model.predict_player_scorer_with_match_over_under(
                    home_player_mean, home_mean, home_mean, away_mean, threshold, True, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if home_goal_count_index > 0:
                    home_match_ou = self._adjust_market_for_nplus(home_match_ou, home_scorer_1plus, home_scorer_prob)
                
                # Set table items for this threshold
                if row < self.home_match_ou_table.rowCount():
                    self.set_table_item(self.home_match_ou_table, row, home_match_ou['scores_over'], f"Scores & Over {threshold}", home_goal_count_index)
                    self.set_table_item(self.home_match_ou_table, row+1, home_match_ou['scores_under'], f"Scores & Under {threshold}", home_goal_count_index)
                row += 2
                
            # Away player Match Over/Under
            row = 0
            for threshold in thresholds:
                # Get the match over/under result for this specific threshold
                away_match_ou = model.predict_player_scorer_with_match_over_under(
                    away_player_mean, away_mean, home_mean, away_mean, threshold, False, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if away_goal_count_index > 0:
                    away_match_ou = self._adjust_market_for_nplus(away_match_ou, away_scorer_1plus, away_scorer_prob)
                
                # Set table items for this threshold
                if row < self.away_match_ou_table.rowCount():
                    self.set_table_item(self.away_match_ou_table, row, away_match_ou['scores_over'], f"Scores & Over {threshold}", away_goal_count_index)
                    self.set_table_item(self.away_match_ou_table, row+1, away_match_ou['scores_under'], f"Scores & Under {threshold}", away_goal_count_index)
                row += 2
            
            # Populate Home Team Over/Under tables
            # Home player Home Team O/U
            row = 0
            for threshold in thresholds:
                # Get the home team over/under result for this specific threshold
                home_team_ou = model.predict_player_scorer_with_home_team_over_under(
                    home_player_mean, home_mean, home_mean, away_mean, threshold, True, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if home_goal_count_index > 0:
                    home_team_ou = self._adjust_market_for_nplus(home_team_ou, home_scorer_1plus, home_scorer_prob)
                
                # Set table items for this threshold
                if row < self.home_team_ou_table.rowCount():
                    self.set_table_item(self.home_team_ou_table, row, home_team_ou['scores_over'], f"Scores & Home Over {threshold}", home_goal_count_index)
                    self.set_table_item(self.home_team_ou_table, row+1, home_team_ou['scores_under'], f"Scores & Home Under {threshold}", home_goal_count_index)
                row += 2
            
            # Away player Home Team O/U
            row = 0
            for threshold in thresholds:
                # Get the home team over/under result for this specific threshold
                away_home_ou = model.predict_player_scorer_with_home_team_over_under(
                    away_player_mean, away_mean, home_mean, away_mean, threshold, False, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if away_goal_count_index > 0:
                    away_home_ou = self._adjust_market_for_nplus(away_home_ou, away_scorer_1plus, away_scorer_prob)
                
                # Set table items for this threshold
                if row < self.away_home_ou_table.rowCount():
                    self.set_table_item(self.away_home_ou_table, row, away_home_ou['scores_over'], f"Scores & Home Over {threshold}", away_goal_count_index)
                    self.set_table_item(self.away_home_ou_table, row+1, away_home_ou['scores_under'], f"Scores & Home Under {threshold}", away_goal_count_index)
                row += 2
            
            # Populate Away Team Over/Under tables
            # Home player Away Team O/U
            row = 0
            for threshold in thresholds:
                # Get the away team over/under result for this specific threshold
                home_away_ou = model.predict_player_scorer_with_away_team_over_under(
                    home_player_mean, home_mean, home_mean, away_mean, threshold, True, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if home_goal_count_index > 0:
                    home_away_ou = self._adjust_market_for_nplus(home_away_ou, home_scorer_1plus, home_scorer_prob)
                
                # Set table items for this threshold
                if row < self.home_away_ou_table.rowCount():
                    self.set_table_item(self.home_away_ou_table, row, home_away_ou['scores_over'], f"Scores & Away Over {threshold}", home_goal_count_index)
                    self.set_table_item(self.home_away_ou_table, row+1, home_away_ou['scores_under'], f"Scores & Away Under {threshold}", home_goal_count_index)
                row += 2
            
            # Away player Away Team O/U
            row = 0
            for threshold in thresholds:
                # Get the away team over/under result for this specific threshold
                away_away_ou = model.predict_player_scorer_with_away_team_over_under(
                    away_player_mean, away_mean, home_mean, away_mean, threshold, False, rho, score_grid=score_grid
                )
                # Adjust for 2+/3+ if needed
                if away_goal_count_index > 0:
                    away_away_ou = self._adjust_market_for_nplus(away_away_ou, away_scorer_1plus, away_scorer_prob)
                
                # Set table items for this threshold
                if row < self.away_team_ou_table.rowCount():
                    self.set_table_item(self.away_team_ou_table, row, away_away_ou['scores_over'], f"Scores & Away Over {threshold}", away_goal_count_index)
                    self.set_table_item(self.away_team_ou_table, row+1, away_away_ou['scores_under'], f"Scores & Away Under {threshold}", away_goal_count_index)
                row += 2
            
        except Exception as e:
            print(f"Error calculating BTTS & Over/Under probabilities: {str(e)}")
    
    def _adjust_combinations_for_nplus(self, model, player_mean, team_mean, home_mean, away_mean, is_home_team, rho, n_plus=2):
        """
        Adjusts player scoring combinations for 2+ or 3+ goals
        
        Parameters:
        -----------
        model : DixonColesModel
            The prediction model
        player_mean, team_mean : float
            Expected goals for player and team
        home_mean, away_mean : float
            Expected goals for home and away teams
        is_home_team : bool
            Whether player is on home team
        rho : float
            Correlation parameter
        n_plus : int
            Number of goals (2+ or 3+)
            
        Returns:
        --------
        dict : Adjusted probabilities for player scoring combinations
        """
        # Get regular 1+ scorer result probabilities
        base_result = model.predict_player_scorer_and_result(
            player_mean, team_mean, home_mean, away_mean, is_home_team, rho
        )
        
        # Calculate n+ probability
        if n_plus == 2:
            n_plus_prob = model.predict_player_2plus(player_mean, team_mean)
        elif n_plus == 3:
            n_plus_prob = model.predict_player_3plus(player_mean, team_mean)
        else:
            n_plus_prob = model.predict_player_scorer(player_mean, team_mean)
            
        # Calculate anytime scorer probability
        anytime_prob = model.predict_player_scorer(player_mean, team_mean)
        
        # If anytime prob is zero, avoid division by zero
        if anytime_prob <= 0:
            return {
                'scores_team_wins': 0,
                'scores_team_draws': 0,
                'scores_team_loses': 0,
                'no_score_team_wins': 0,
                'no_score_team_draws': 0,
                'no_score_team_loses': 0
            }
        
        # Adjust probabilities based on n+ to 1+ ratio
        ratio = n_plus_prob / anytime_prob
        
        return {
            'scores_team_wins': base_result['scores_team_wins'] * ratio,
            'scores_team_draws': base_result['scores_team_draws'] * ratio,
            'scores_team_loses': base_result['scores_team_loses'] * ratio,
            'no_score_team_wins': base_result['no_score_team_wins'],
            'no_score_team_draws': base_result['no_score_team_draws'],
            'no_score_team_loses': base_result['no_score_team_loses']
        }
    
    def _adjust_market_for_nplus(self, market_dict, anytime_prob, goal_prob):
        """
        Adjusts a market dictionary for n+ goals
        
        Parameters:
        -----------
        market_dict : dict
            Market probabilities to adjust
        anytime_prob : float
            Probability of scoring anytime (1+)
        goal_prob : float
            Probability of scoring n+ goals
            
        Returns:
        --------
        dict : Adjusted market probabilities
        """
        if anytime_prob <= 0.00001:  # Use a small threshold to avoid division by zero
            return market_dict.copy()
            
        # Calculate adjustment ratio
        ratio = min(goal_prob / anytime_prob, 1.0)  # Ensure ratio never exceeds 1.0
        
        # Find which keys are scoring-related (usually starting with 'scores_')
        scoring_keys = [k for k in market_dict.keys() if k.startswith('scores_')]
        non_scoring_keys = [k for k in market_dict.keys() if not k.startswith('scores_') and k != "team"]
        
        # Create a new dictionary with adjusted values
        adjusted_dict = {}
        
        # Adjust scoring probabilities with the ratio
        total_score_prob = 0
        for key in scoring_keys:
            adjusted_dict[key] = market_dict[key] * ratio
            total_score_prob += adjusted_dict[key]
            
        # Maintain the "team" key if it exists
        if "team" in market_dict:
            adjusted_dict["team"] = market_dict["team"]
            
        # For non-scoring keys, ensure they don't go below the original values
        # This prevents probabilities from becoming inconsistent
        for key in non_scoring_keys:
            adjusted_dict[key] = market_dict[key]
            
        return adjusted_dict
    
    def update_player_scoring_heatmap(self, model, player_mean, home_mean, away_mean, rho, is_home_player, home_team_name, away_team_name, score_grid=None):
        """
        Update the player scoring heatmap visualization
        
        Parameters:
        -----------
        model : DixonColesModel
            The model to use for calculations
        player_mean : float
            Expected goals for the player
        home_mean, away_mean : float
            Expected goals for home and away teams
        rho : float
            Correlation parameter for low-scoring adjustment
        is_home_player : bool
            True if player is on home team, False otherwise
        home_team_name, away_team_name : str
            Names of home and away teams for display
        score_grid : np.array, optional
            Pre-calculated Dixon-Coles score grid (if not provided, will calculate)
        """
        # Get team mean for player's team
        team_mean = home_mean if is_home_player else away_mean
        
        # Get max goals from parent window
        main_window = self.window()
        max_goals = main_window.max_goals_spin.value()
        
        # Get player's contribution ratio (how much of team's goals this player scores)
        player_ratio = player_mean / team_mean if team_mean > 0 else 0
        
        # Get the player name
        player_name = self.home_player_name.text() if is_home_player else self.away_player_name.text()
        if not player_name:
            player_name = f"{home_team_name if is_home_player else away_team_name} Player"
        
        # Create grid to store player scoring probabilities for each scoreline
        player_grid = np.zeros((max_goals + 1, max_goals + 1))
        
        # If score_grid is not provided, calculate it
        if score_grid is None:
            score_grid = model.predict_score_grid(home_mean, away_mean, max_goals, rho)
            
        # Calculate player scoring probability for each scoreline using the provided grid
        for i in range(max_goals + 1):  # home goals
            for j in range(max_goals + 1):  # away goals
                # Use the pre-calculated scoreline probability
                scoreline_prob = score_grid[i, j]
                
                # Team goals scored by player's team
                team_goals = i if is_home_player else j
                
                # Calculate probability that player scores at least once given team scored team_goals
                if team_goals == 0:
                    prob_player_scores = 0  # Player can't score if team doesn't score
                else:
                    # Probability player scores at least one of the team's goals
                    prob_player_scores = 1 - (1 - player_ratio) ** team_goals
                
                # Probability of scoreline AND player scores
                player_grid[i, j] = scoreline_prob * prob_player_scores
        
        # Normalize the grid
        total_scoring_prob = np.sum(player_grid)
        if total_scoring_prob > 0:
            player_grid = player_grid / total_scoring_prob
            
        # Get the appropriate canvas
        canvas = self.home_heatmap_canvas if is_home_player else self.away_heatmap_canvas
        
        # Create the player scoring heatmap
        create_player_scoring_heatmap_for_canvas(
            canvas, 
            player_grid, 
            max_goals=max_goals,
            home_team=home_team_name,
            away_team=away_team_name,
            player_name=player_name
        )
    
    def update_player_names(self):
        """Update player names in group box titles"""
        home_name = self.home_player_name.text()
        away_name = self.away_player_name.text()
        
        # Find the parent QGroupBox widgets and update their titles
        for widget in self.findChildren(QGroupBox):
            if "Home Player" in widget.title():
                widget.setTitle(f"Home Player ({home_name})")
            elif "Away Player" in widget.title():
                widget.setTitle(f"Away Player ({away_name})")
    
    def set_table_item(self, table, row, prob, market, goal_count_index=0):
        """Set a table cell with probability and odds"""
        # Adjust market text based on goal count selection
        if goal_count_index > 0 and "Scores & Team" in market:
            # For combination markets, prefix with the goal count
            if goal_count_index == 1:  # 2+ goals
                market = market.replace("Scores", "Scores 2+ Goals")
            elif goal_count_index == 2:  # 3+ goals
                market = market.replace("Scores", "Scores 3+ Goals")
                
        # Calculate odds
        odds_value = probability_to_decimal_odds(prob)
        
        # Create a button for the table cell
        btn = QPushButton(f"{market}\n{odds_value:.2f}")
        
        # Style the button to match betting markets widget
        btn.setStyleSheet("""
            background-color: #2d7dd2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 16px 12px 16px;
            font-weight: bold;
            text-align: center;
            min-width: 150px;
            min-height: 60px;
        """)
        
        # Add the button to a container
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(btn)
        layout.setContentsMargins(1, 1, 1, 1)
        container.setLayout(layout)
        
        # Set the widget in the table
        table.setCellWidget(row, 0, container)
        
        # Set the odds column
        odds_item = QTableWidgetItem(f"{odds_value:.2f}")
        odds_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        table.setItem(row, 1, odds_item)
        
        # Set the probability column
        prob_item = QTableWidgetItem(f"{prob:.4f}")
        prob_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        table.setItem(row, 2, prob_item)
    
    def set_ou_table_item(self, table, row, prob, market):
        """Set an over/under table cell with probability and odds"""
        # Calculate over probability and odds
        over_prob = prob["over"]
        over_odds = probability_to_decimal_odds(over_prob)
        
        # Create over button
        over_btn = QPushButton(f"Over {market.split()[-1]}\n{over_odds:.2f}")
        over_btn.setStyleSheet("""
            background-color: #2d7dd2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 16px 12px 16px;
            font-weight: bold;
            text-align: center;
            min-width: 120px;
            min-height: 50px;
        """)
        
        # Add the button to a layout
        over_container = QWidget()
        over_layout = QVBoxLayout(over_container)
        over_layout.addWidget(over_btn)
        over_layout.setContentsMargins(1, 1, 1, 1)
        over_container.setLayout(over_layout)
        
        # Set the over widget
        table.setCellWidget(row, 1, over_container)
        
        # Set over odds
        over_odds_item = QTableWidgetItem(f"{over_odds:.2f}")
        over_odds_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        table.setItem(row, 2, over_odds_item)
        
        # Set over probability
        over_prob_item = QTableWidgetItem(f"{over_prob:.4f}")
        over_prob_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        table.setItem(row, 3, over_prob_item)
        
        # Calculate under probability and odds
        under_prob = prob["under"]
        under_odds = probability_to_decimal_odds(under_prob)
        
        # Create under button
        under_btn = QPushButton(f"Under {market.split()[-1]}\n{under_odds:.2f}")
        under_btn.setStyleSheet("""
            background-color: #2d7dd2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 16px 12px 16px;
            font-weight: bold;
            text-align: center;
            min-width: 120px;
            min-height: 50px;
        """)
        
        # Add the button to a layout
        under_container = QWidget()
        under_layout = QVBoxLayout(under_container)
        under_layout.addWidget(under_btn)
        under_layout.setContentsMargins(1, 1, 1, 1)
        under_container.setLayout(under_layout)
        
        # Set the under widget
        table.setCellWidget(row, 4, under_container)
        
        # Set under odds
        under_odds_item = QTableWidgetItem(f"{under_odds:.2f}")
        under_odds_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        table.setItem(row, 5, under_odds_item)
        
        # Set under probability
        under_prob_item = QTableWidgetItem(f"{under_prob:.4f}")
        under_prob_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        table.setItem(row, 6, under_prob_item)
    
    def set_exact_goals_item(self, table, row, prob):
        """Set an exact goals table cell with probability and odds"""
        # Calculate odds
        odds_value = probability_to_decimal_odds(prob)
        
        # Create button
        btn = QPushButton(f"Exactly {row} Goal{'s' if row != 1 else ''}\n{odds_value:.2f}")
        btn.setStyleSheet("""
            background-color: #2d7dd2;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 4px 16px 12px 16px;
            font-weight: bold;
            text-align: center;
            min-width: 160px;
            min-height: 50px;
        """)
        
        # Add the button to a layout
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(btn)
        layout.setContentsMargins(1, 1, 1, 1)
        container.setLayout(layout)
        
        # Set the widget in the table
        table.setCellWidget(row, 0, container)
        
        # Set the odds column
        odds_item = QTableWidgetItem(f"{odds_value:.2f}")
        odds_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        table.setItem(row, 1, odds_item)
        
        # Set the probability column
        prob_item = QTableWidgetItem(f"{prob:.4f}")
        prob_item.setTextAlignment(Qt.AlignCenter | Qt.AlignVCenter)
        table.setItem(row, 2, prob_item)
    
    def populate_ou_tables(self, player_ou_probabilities, home_team, away_team):
        """Populate the over/under tables for a player"""
        thresholds = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
        
        # Get the tables based on whether it's a home or away player
        if player_ou_probabilities["team"] == home_team:
            match_ou_table = self.home_match_ou_table
            team_ou_table = self.home_team_ou_table
            away_ou_table = self.home_away_ou_table
        else:
            match_ou_table = self.away_match_ou_table
            team_ou_table = self.away_team_ou_table
            away_ou_table = self.away_home_ou_table

        # Clear tables
        for table in [match_ou_table, team_ou_table, away_ou_table]:
            for row in range(table.rowCount()):
                for col in range(table.columnCount()):
                    if table.item(row, col):
                        table.takeItem(row, col)
                    widget = table.cellWidget(row, col)
                    if widget:
                        table.removeCellWidget(row, col)

        # Populate match over/under
        for i, threshold in enumerate(thresholds):
            # Over threshold
            over_key = f"player_scores_and_match_over_{threshold}"
            if over_key in player_ou_probabilities:
                prob = player_ou_probabilities[over_key]
                odds = probability_to_decimal_odds(prob)
                
                # Create over button
                over_btn = QPushButton(f"Scores & O{threshold}\n{odds:.2f}")
                over_btn.setStyleSheet("""
                    background-color: #2d7dd2;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 16px 12px 16px;
                    font-weight: bold;
                    text-align: center;
                """)
                
                # Add the button to a layout
                over_container = QWidget()
                over_layout = QVBoxLayout(over_container)
                over_layout.addWidget(over_btn)
                over_layout.setContentsMargins(0, 0, 0, 0)
                over_container.setLayout(over_layout)
                
                # Set the over widget
                match_ou_table.setCellWidget(i*2, 0, over_container)
                
                # Set probability
                prob_item = QTableWidgetItem(f"{prob:.5f}")
                prob_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                match_ou_table.setItem(i*2, 2, prob_item)
            
            # Under threshold
            under_key = f"player_scores_and_match_under_{threshold}"
            if under_key in player_ou_probabilities:
                prob = player_ou_probabilities[under_key]
                odds = probability_to_decimal_odds(prob)
                
                # Create under button
                under_btn = QPushButton(f"Scores & U{threshold}\n{odds:.2f}")
                under_btn.setStyleSheet("""
                    background-color: #2d7dd2;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 16px 12px 16px;
                    font-weight: bold;
                    text-align: center;
                """)
                
                # Add the button to a layout
                under_container = QWidget()
                under_layout = QVBoxLayout(under_container)
                under_layout.addWidget(under_btn)
                under_layout.setContentsMargins(0, 0, 0, 0)
                under_container.setLayout(under_layout)
                
                # Set the under widget
                match_ou_table.setCellWidget(i*2+1, 0, under_container)
                
                # Set probability
                prob_item = QTableWidgetItem(f"{prob:.5f}")
                prob_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                match_ou_table.setItem(i*2+1, 2, prob_item)

        # Populate home team over/under
        for i, threshold in enumerate(thresholds):
            # Over threshold
            over_key = f"player_scores_and_home_over_{threshold}"
            if over_key in player_ou_probabilities:
                prob = player_ou_probabilities[over_key]
                odds = probability_to_decimal_odds(prob)
                
                # Create over button
                over_btn = QPushButton(f"Scores & Home O{threshold}\n{odds:.2f}")
                over_btn.setStyleSheet("""
                    background-color: #2d7dd2;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 16px 12px 16px;
                    font-weight: bold;
                    text-align: center;
                """)
                
                # Add the button to a layout
                over_container = QWidget()
                over_layout = QVBoxLayout(over_container)
                over_layout.addWidget(over_btn)
                over_layout.setContentsMargins(0, 0, 0, 0)
                over_container.setLayout(over_layout)
                
                # Set the over widget
                team_ou_table.setCellWidget(i*2, 0, over_container)
                
                # Set probability
                prob_item = QTableWidgetItem(f"{prob:.5f}")
                prob_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                team_ou_table.setItem(i*2, 2, prob_item)
            
            # Under threshold
            under_key = f"player_scores_and_home_under_{threshold}"
            if under_key in player_ou_probabilities:
                prob = player_ou_probabilities[under_key]
                odds = probability_to_decimal_odds(prob)
                
                # Create under button
                under_btn = QPushButton(f"Scores & Home U{threshold}\n{odds:.2f}")
                under_btn.setStyleSheet("""
                    background-color: #2d7dd2;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 16px 12px 16px;
                    font-weight: bold;
                    text-align: center;
                """)
                
                # Add the button to a layout
                under_container = QWidget()
                under_layout = QVBoxLayout(under_container)
                under_layout.addWidget(under_btn)
                under_layout.setContentsMargins(0, 0, 0, 0)
                under_container.setLayout(under_layout)
                
                # Set the under widget
                team_ou_table.setCellWidget(i*2+1, 0, under_container)
                
                # Set probability
                prob_item = QTableWidgetItem(f"{prob:.5f}")
                prob_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                team_ou_table.setItem(i*2+1, 2, prob_item)

        # Populate away team over/under
        for i, threshold in enumerate(thresholds):
            # Over threshold
            over_key = f"player_scores_and_away_over_{threshold}"
            if over_key in player_ou_probabilities:
                prob = player_ou_probabilities[over_key]
                odds = probability_to_decimal_odds(prob)
                
                # Create over button
                over_btn = QPushButton(f"Scores & Away O{threshold}\n{odds:.2f}")
                over_btn.setStyleSheet("""
                    background-color: #2d7dd2;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 16px 12px 16px;
                    font-weight: bold;
                    text-align: center;
                """)
                
                # Add the button to a layout
                over_container = QWidget()
                over_layout = QVBoxLayout(over_container)
                over_layout.addWidget(over_btn)
                over_layout.setContentsMargins(0, 0, 0, 0)
                over_container.setLayout(over_layout)
                
                # Set the over widget
                away_ou_table.setCellWidget(i*2, 0, over_container)
                
                # Set probability
                prob_item = QTableWidgetItem(f"{prob:.5f}")
                prob_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                away_ou_table.setItem(i*2, 2, prob_item)
            
            # Under threshold
            under_key = f"player_scores_and_away_under_{threshold}"
            if under_key in player_ou_probabilities:
                prob = player_ou_probabilities[under_key]
                odds = probability_to_decimal_odds(prob)
                
                # Create under button
                under_btn = QPushButton(f"Scores & Away U{threshold}\n{odds:.2f}")
                under_btn.setStyleSheet("""
                    background-color: #2d7dd2;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    padding: 4px 16px 12px 16px;
                    font-weight: bold;
                    text-align: center;
                """)
                
                # Add the button to a layout
                under_container = QWidget()
                under_layout = QVBoxLayout(under_container)
                under_layout.addWidget(under_btn)
                under_layout.setContentsMargins(0, 0, 0, 0)
                under_container.setLayout(under_layout)
                
                # Set the under widget
                away_ou_table.setCellWidget(i*2+1, 0, under_container)
                
                # Set probability
                prob_item = QTableWidgetItem(f"{prob:.5f}")
                prob_item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                away_ou_table.setItem(i*2+1, 2, prob_item)

    def show_expanded_player_heatmap(self, is_home_player):
        """
        Show an expanded player scoring heatmap in a separate window
        
        Parameters:
        -----------
        is_home_player : bool
            True if showing home player, False if away player
        """
        # Get parameters
        main_window = self.window()
        home_mean = main_window.home_mean_spin.value()
        away_mean = main_window.away_mean_spin.value()
        max_goals = main_window.max_goals_spin.value()
        rho = main_window.rho_spin.value()
        model = main_window.model
        
        # Calculate score grid once for efficiency
        score_grid = model.predict_score_grid(home_mean, away_mean, max_goals, rho)
        
        # Get player data
        if is_home_player:
            player_name = self.home_player_name.text()
            if not player_name:
                player_name = f"{main_window.home_team_name.text()} Player"
            player_mean = self.get_home_player_xg(home_mean)
            team_mean = home_mean
        else:
            player_name = self.away_player_name.text()
            if not player_name:
                player_name = f"{main_window.away_team_name.text()} Player"
            player_mean = self.get_away_player_xg(away_mean)
            team_mean = away_mean
            
        # Create a new dialog
        dialog = QDialog(self)
        dialog.setWindowTitle(f"{player_name} Scoring Heatmap")
        dialog.setMinimumSize(800, 600)
        
        # Create layout
        layout = QVBoxLayout(dialog)
        
        # Create title label
        title = QLabel(f"{player_name} Scoring Probabilities by Scoreline")
        title.setStyleSheet("font-size: 16px; font-weight: bold;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Create canvas for heatmap
        canvas = MatplotlibCanvas(width=10, height=8)
        layout.addWidget(canvas)
        
        # Get player contribution ratio
        player_ratio = player_mean / team_mean if team_mean > 0 else 0
        
        # Create grid to store player scoring probabilities for each scoreline
        player_grid = np.zeros((max_goals + 1, max_goals + 1))
        
        # Calculate player scoring probabilities using the pre-calculated score grid
        for i in range(max_goals + 1):  # home goals
            for j in range(max_goals + 1):  # away goals
                # Use the pre-calculated scoreline probability
                scoreline_prob = score_grid[i, j]
                
                # Team goals scored by player's team
                team_goals = i if is_home_player else j
                
                # Calculate probability that player scores at least once given team scored team_goals
                if team_goals == 0:
                    prob_player_scores = 0  # Player can't score if team doesn't score
                else:
                    # Probability player scores at least one of the team's goals
                    prob_player_scores = 1 - (1 - player_ratio) ** team_goals
                
                # Probability of scoreline AND player scores
                player_grid[i, j] = scoreline_prob * prob_player_scores
        
        # Normalize the grid
        total_scoring_prob = np.sum(player_grid)
        if total_scoring_prob > 0:
            player_grid = player_grid / total_scoring_prob
        
        # Create the player scoring heatmap
        create_player_scoring_heatmap_for_canvas(
            canvas, 
            player_grid, 
            max_goals=max_goals,
            home_team=main_window.home_team_name.text(),
            away_team=main_window.away_team_name.text(),
            player_name=player_name
        )
        
        # Add close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dialog.close)
        layout.addWidget(close_btn)
        
        # Show the dialog
        dialog.exec_()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Set application style and colors
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
                border-radius: 4px;
            }
            QTabBar::tab {
                background-color: #e6e6e6;
                border: 1px solid #cccccc;
                border-bottom-color: #cccccc;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: white;
                border-bottom-color: white;
            }
            QPushButton {
                background-color: #2d7dd2;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 4px 16px 12px 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1e6cba;
            }
            QPushButton:pressed {
                background-color: #185a9d;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
                padding: 6px;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 12px;
                padding-top: 20px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }
            QLabel {
                color: #333333;
            }
            QSlider::groove:horizontal {
                border: 1px solid #cccccc;
                height: 8px;
                background: #f0f0f0;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2d7dd2;
                border: 1px solid #1e6cba;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)

        # Set up the main window
        self.setWindowTitle("Dixon-Coles Soccer Prediction Model")
        self.setMinimumSize(1200, 900)
        
        # Initialize model
        self.model = DixonColesModel(max_goals=10)
        
        # Set up the central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # Create the prediction tab
        self.prediction_tab = QWidget()
        self.tabs.addTab(self.prediction_tab, "Score Prediction")
        
        # Create the betting tab
        self.betting_tab = QWidget()
        self.tabs.addTab(self.betting_tab, "Betting Markets")
        
        # Create the player scoring tab
        self.player_tab = QWidget()
        self.tabs.addTab(self.player_tab, "Player Scoring")
        
        # Set up the tab layouts
        self.setup_prediction_tab()
        self.setup_betting_tab()
        self.setup_player_tab()
        
        # Create a global calculate button
        self.calc_button = QPushButton("Calculate")
        self.calc_button.setStyleSheet("""
            font-size: 18px; 
            padding: 12px 24px;
            background-color: #28a745;
        """)
        self.calc_button.clicked.connect(self.update_prediction)
        self.main_layout.addWidget(self.calc_button)
        
        # Initial calculation with default values
        self.update_prediction()
    
    def setup_prediction_tab(self):
        """Set up the UI elements for prediction"""
        layout = QVBoxLayout(self.prediction_tab)
        
        # Create title label
        title_label = QLabel("Dixon-Coles Soccer Prediction Model")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2d7dd2;
            margin: 10px 0;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Create input controls section
        input_group = QGroupBox("Expected Goals Input")
        input_layout = QFormLayout(input_group)
        
        # Home team mean goals
        self.home_mean_spin = QDoubleSpinBox()
        self.home_mean_spin.setRange(0.1, 5.0)
        self.home_mean_spin.setSingleStep(0.1)
        self.home_mean_spin.setDecimals(5)
        self.home_mean_spin.setValue(1.5)
        input_layout.addRow("Home Team Mean Goals:", self.home_mean_spin)
        
        # Away team mean goals
        self.away_mean_spin = QDoubleSpinBox()
        self.away_mean_spin.setRange(0.1, 5.0)
        self.away_mean_spin.setSingleStep(0.1)
        self.away_mean_spin.setDecimals(5)
        self.away_mean_spin.setValue(1.2)
        input_layout.addRow("Away Team Mean Goals:", self.away_mean_spin)
        
        # Maximum display goals
        self.max_goals_spin = QSpinBox()
        self.max_goals_spin.setRange(1, 10)
        self.max_goals_spin.setValue(5)
        input_layout.addRow("Max Display Goals:", self.max_goals_spin)
        
        # Dixon-Coles correlation parameter (rho) - Now a float input
        self.rho_spin = QDoubleSpinBox()
        self.rho_spin.setRange(-0.2, 0.2)
        self.rho_spin.setSingleStep(0.01)
        self.rho_spin.setDecimals(2)
        self.rho_spin.setValue(-0.16)
        input_layout.addRow("Low Score Correlation (rho):", self.rho_spin)
        
        # Add team names (optional, just for display)
        home_team_name = QLineEdit("Home Team")
        self.home_team_name = home_team_name
        input_layout.addRow("Home Team Name:", home_team_name)
        
        away_team_name = QLineEdit("Away Team")
        self.away_team_name = away_team_name
        input_layout.addRow("Away Team Name:", away_team_name)
        
        # Add buttons for preset values
        preset_layout = QHBoxLayout()
        
        low_scoring_btn = QPushButton("Low Scoring")
        low_scoring_btn.clicked.connect(lambda: self.set_preset(0.8, 0.7))
        preset_layout.addWidget(low_scoring_btn)
        
        average_btn = QPushButton("Average")
        average_btn.clicked.connect(lambda: self.set_preset(1.5, 1.2))
        preset_layout.addWidget(average_btn)
        
        high_scoring_btn = QPushButton("High Scoring")
        high_scoring_btn.clicked.connect(lambda: self.set_preset(2.2, 1.8))
        preset_layout.addWidget(high_scoring_btn)
        
        input_layout.addRow("Presets:", preset_layout)
        
        # Add the input group to the main layout
        layout.addWidget(input_group)
        
        # Results section
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout(results_group)
        
        # Result summary section
        self.result_summary = QLabel()
        self.result_summary.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        self.result_summary.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.result_summary)
        
        # Over/Under summary section
        self.over_under_summary = QLabel()
        self.over_under_summary.setStyleSheet("font-size: 14px; padding: 5px;")
        self.over_under_summary.setAlignment(Qt.AlignCenter)
        results_layout.addWidget(self.over_under_summary)
        
        # Create MatplotlibCanvas for the heatmap
        self.heatmap_canvas = MatplotlibCanvas(width=10, height=8)
        results_layout.addWidget(self.heatmap_canvas)
        
        # Add results group to the main layout
        layout.addWidget(results_group)
    
    def setup_betting_tab(self):
        """Set up the UI elements for betting markets"""
        layout = QVBoxLayout(self.betting_tab)
        
        # Create title label
        title_label = QLabel("Betting Markets")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2d7dd2;
            margin: 10px 0;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Add betting markets widget
        self.betting_widget = BettingMarketWidget()
        
        # Wrap the betting widget in a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.betting_widget)
        
        layout.addWidget(scroll_area)
    
    def setup_player_tab(self):
        """Set up the player scoring tab"""
        layout = QVBoxLayout(self.player_tab)
        
        # Create title label
        title_label = QLabel("Player Scoring Probabilities")
        title_label.setStyleSheet("""
            font-size: 24px;
            font-weight: bold;
            color: #2d7dd2;
            margin: 10px 0;
        """)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)
        
        # Add player scoring widget
        self.player_widget = PlayerScoringWidget()
        
        # Wrap the player widget in a scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.player_widget)
        
        layout.addWidget(scroll_area)
    
    def set_preset(self, home_value, away_value):
        """Set preset values for mean goals"""
        self.home_mean_spin.setValue(home_value)
        self.away_mean_spin.setValue(away_value)
        # No need to call update_prediction here since we only update on button click
    
    def update_prediction(self):
        """Update the prediction visualization based on current inputs"""
        try:
            # Get input values
            home_mean = self.home_mean_spin.value()
            away_mean = self.away_mean_spin.value()
            max_goals = self.max_goals_spin.value()
            rho = self.rho_spin.value()
            
            # Get team names for display
            home_team = self.home_team_name.text()
            away_team = self.away_team_name.text()
            
            # Get prediction grid - calculate once for efficiency
            score_grid = self.model.predict_score_grid(home_mean, away_mean, max_goals, rho)
            
            # Update the heatmap
            create_heatmap_for_canvas(
                self.heatmap_canvas, 
                score_grid, 
                max_goals=max_goals,
                home_team=home_team,
                away_team=away_team
            )
            
            # Update the results summary
            outcomes = self.model.predict_match_outcome(home_mean, away_mean, rho)
            self.result_summary.setText(format_outcome_probabilities(outcomes))
            
            # Update over/under summary
            over_under = self.model.predict_over_under(home_mean, away_mean, 2.5, rho)
            self.over_under_summary.setText(format_over_under(over_under, 2.5))
            
            # Update betting markets
            self.betting_widget.update_odds(self.model, home_mean, away_mean, rho)
            
            # Update player scoring probabilities - pass the pre-calculated score grid
            self.player_widget.update_player_odds(self.model, home_mean, away_mean, rho, score_grid)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error updating prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            
# PlayerScoringWidget class is already defined above at line 246