from blackhc.mdp import dsl

# Hamid Maei's Counterexample
def counterexample_env():
    with dsl.new() as mdp:
        s1 = dsl.state()
        s2 = dsl.state()

        a1 = dsl.action()
        a2 = dsl.action()

        (s1 | s2) & a1 > s2
        (s1 | s2) & a2 > s1

        (s1 | s2) & a1 > dsl.reward(+1)
        (s1 | s2) & a2 > dsl.reward(0)

        return mdp.validate()

# Our Counterexample
def new_counterexample_env():
    with dsl.new() as mdp:
        s1 = dsl.state()
        s2 = dsl.state()
        s3 = dsl.state()
        s4 = dsl.state()
        s5 = dsl.terminal_state()

        l = dsl.action()
        r = dsl.action()

        #

        s1 & l > s2
        s1 & r > s4

        s2 & l > s3
        s2 & r > s1

        s3 & l > s5
        s3 & r > s2

        s4 & l > s1
        s4 & r > s5

        #

        s1 & l > dsl.reward(-1)
        s1 & r > dsl.reward(-1)

        s2 & l > dsl.reward(-1)
        s2 & r > dsl.reward(-1)

        s3 & l > dsl.reward(-1)
        s3 & r > dsl.reward(-1)

        s4 & l > dsl.reward(-1)
        s4 & r > dsl.reward(-1)

        return mdp.validate()

# My tiny counterexample
def tiny_counterexample_env():
    with dsl.new() as mdp:
        s0 = dsl.state()
        s1 = dsl.state()
        s2 = dsl.state()
        s3 = dsl.terminal_state()

        a0 = dsl.action()
        a1 = dsl.action()

        #

        s0 & a0 > s1
        s0 & a1 > s2

        s1 & a0 > s3
        s1 & a1 > s3

        s2 & a0 > s3
        s2 & a1 > s3
        #

        s0 & a0 > dsl.reward(0)
        s0 & a1 > dsl.reward(0)

        s1 & a0 > dsl.reward(2)
        s1 & a1 > dsl.reward(0)

        s2 & a0 > dsl.reward(0)
        s2 & a1 > dsl.reward(1)

        return mdp.validate()


# Long counterexample
def long_counterexample_env(middle_steps=1):
    with dsl.new() as mdp:
        s0 = dsl.state()

        a0 = dsl.action()
        a1 = dsl.action()

        middle_l = [s0, dsl.state()]
        middle_r = [s0, dsl.state()]

        s0 & a0 > middle_l[-1]
        s0 & a1 > middle_r[-1]

        s0 & a0 > dsl.reward(0)
        s0 & a1 > dsl.reward(0)


        for step in range(middle_steps - 1):
            middle_l.append(dsl.state())
            middle_l[-2] & a0 > middle_l[-1]
            middle_l[-2] & a1 > middle_l[-3]
            middle_l[-2] & a0 > dsl.reward(0)
            middle_l[-2] & a1 > dsl.reward(0)

            middle_r.append(dsl.state())
            middle_r[-2] & a0 > middle_r[-3]
            middle_r[-2] & a1 > middle_r[-1]
            middle_r[-2] & a0 > dsl.reward(0)
            middle_r[-2] & a1 > dsl.reward(0)

        s3 = dsl.terminal_state()

        #

        middle_l[-1] & a0 > s3
        middle_l[-1] & a1 > s3

        middle_r[-1] & a0 > s3
        middle_r[-1] & a1 > s3
        #


        middle_l[-1] & a0 > dsl.reward(2)
        middle_l[-1] & a1 > dsl.reward(0)

        middle_r[-1] & a0 > dsl.reward(0)
        middle_r[-1] & a1 > dsl.reward(1)

        return mdp.validate()
