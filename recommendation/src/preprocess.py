def build_dimension_scores(df):
    df = df.copy()

    df['attacking_score'] = (
        df['xg_p90'] + df['shots_p90'] + df['key_passes_p90']
    ) / 3

    df['progression_score'] = (
        df['pass_accuracy'] + df['carries_p90'] + df['dribble_success_rate']
    ) / 3

    df['defensive_score'] = (
        df['interceptions_p90'] + df['duel_win_rate'] + df['recoveries_p90']
    ) / 3

    df['discipline_score'] = 0

    df['spatial_score'] = (
        df['mean_x'] + df['mean_y'] + df['activity_spread']
    ) / 3

    return df