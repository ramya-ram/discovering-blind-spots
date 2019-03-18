from gym.envs.registration import register

for game in ['SourceCatcher', 'TargetCatcher', 'SourceFlappyBird', 'TargetFlappyBird']:
    nondeterministic = False
    register(
        id='{}-v0'.format(game),
        entry_point='domains.ple.ple_env:PLEEnv',
        kwargs={'prespecified_game':False, 'game_name': game, 'display_screen':False},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
        nondeterministic=nondeterministic,
    )