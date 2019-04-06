def get_payoff(model_buzz_pos, model_correct, player_buzz_pos, player_correct):
    if player_buzz_pos < model_buzz_pos :
        if player_correct :
            payoff = -10
        elif model_correct:
            payoff = 15
        else:
            payoff = 5
    elif player_buzz_pos > model_buzz_pos:
        if model_correct:
            payoff = 10
        elif player_correct:
            payoff = -15
        else:
            payoff = -5
    else:
        if model_correct:
            if player_correct:
                payoff = 0
            else:
                payoff = 15
        else:
            if player_correct:
                payoff = -15
            else:
                payoff = 0
    return payoff

