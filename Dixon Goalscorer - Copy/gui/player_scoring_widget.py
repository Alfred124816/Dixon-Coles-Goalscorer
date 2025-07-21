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
                table.setItem(row, col, QTableWidgetItem(""))

    # Populate match over/under
    for i, threshold in enumerate(thresholds):
        # Over threshold
        over_key = f"player_scores_and_match_over_{threshold}"
        if over_key in player_ou_probabilities:
            prob = player_ou_probabilities[over_key]
            odds = probability_to_decimal_odds(prob)
            match_ou_table.setItem(i*2, 0, QTableWidgetItem(f"Scores & O{threshold}"))
            match_ou_table.setItem(i*2, 1, QTableWidgetItem(f"{odds:.2f}"))
            match_ou_table.setItem(i*2, 2, QTableWidgetItem(f"{prob:.2%}"))
        
        # Under threshold
        under_key = f"player_scores_and_match_under_{threshold}"
        if under_key in player_ou_probabilities:
            prob = player_ou_probabilities[under_key]
            odds = probability_to_decimal_odds(prob)
            match_ou_table.setItem(i*2+1, 0, QTableWidgetItem(f"Scores & U{threshold}"))
            match_ou_table.setItem(i*2+1, 1, QTableWidgetItem(f"{odds:.2f}"))
            match_ou_table.setItem(i*2+1, 2, QTableWidgetItem(f"{prob:.2%}"))

    # Populate home team over/under
    for i, threshold in enumerate(thresholds):
        # Over threshold
        over_key = f"player_scores_and_home_over_{threshold}"
        if over_key in player_ou_probabilities:
            prob = player_ou_probabilities[over_key]
            odds = probability_to_decimal_odds(prob)
            team_ou_table.setItem(i*2, 0, QTableWidgetItem(f"Scores & Home O{threshold}"))
            team_ou_table.setItem(i*2, 1, QTableWidgetItem(f"{odds:.2f}"))
            team_ou_table.setItem(i*2, 2, QTableWidgetItem(f"{prob:.2%}"))
        
        # Under threshold
        under_key = f"player_scores_and_home_under_{threshold}"
        if under_key in player_ou_probabilities:
            prob = player_ou_probabilities[under_key]
            odds = probability_to_decimal_odds(prob)
            team_ou_table.setItem(i*2+1, 0, QTableWidgetItem(f"Scores & Home U{threshold}"))
            team_ou_table.setItem(i*2+1, 1, QTableWidgetItem(f"{odds:.2f}"))
            team_ou_table.setItem(i*2+1, 2, QTableWidgetItem(f"{prob:.2%}"))

    # Populate away team over/under
    for i, threshold in enumerate(thresholds):
        # Over threshold
        over_key = f"player_scores_and_away_over_{threshold}"
        if over_key in player_ou_probabilities:
            prob = player_ou_probabilities[over_key]
            odds = probability_to_decimal_odds(prob)
            away_ou_table.setItem(i*2, 0, QTableWidgetItem(f"Scores & Away O{threshold}"))
            away_ou_table.setItem(i*2, 1, QTableWidgetItem(f"{odds:.2f}"))
            away_ou_table.setItem(i*2, 2, QTableWidgetItem(f"{prob:.2%}"))
        
        # Under threshold
        under_key = f"player_scores_and_away_under_{threshold}"
        if under_key in player_ou_probabilities:
            prob = player_ou_probabilities[under_key]
            odds = probability_to_decimal_odds(prob)
            away_ou_table.setItem(i*2+1, 0, QTableWidgetItem(f"Scores & Away U{threshold}"))
            away_ou_table.setItem(i*2+1, 1, QTableWidgetItem(f"{odds:.2f}"))
            away_ou_table.setItem(i*2+1, 2, QTableWidgetItem(f"{prob:.2%}")) 