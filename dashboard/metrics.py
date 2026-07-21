def calculate_growth_rate(df, column):

    first = df[column].iloc[0]
    last = df[column].iloc[-1]

    growth = ((last-first)/first)*100

    return round(growth,2)



def current_value(df,column):

    return round(
        df[column].iloc[-1],
        2
    )



def p2p_atm_ratio(df):

    ratio = (
        df["p2p_transactions"].sum()
        /
        df["atm_transactions"].sum()
    )

    return round(ratio,2)



def target_progress(current,target):

    return round(
        (current/target)*100,
        2
    )