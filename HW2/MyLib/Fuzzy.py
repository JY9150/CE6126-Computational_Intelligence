from abc import ABC, abstractmethod
import numpy as np
from matplotlib import pyplot as plt
import math

class FuzzySet():
    def __init__(self, min_x: float, max_x: float, membership_function: list[float], name: str = "", id: str = "") -> None:
        self.min_x = min_x
        self.max_x = max_x
        self.membership_function = membership_function
        self.name = name
        self.id = id
        
    @property
    def number_of_points(self):
        return len(self.membership_function)
    
    @property
    def height(self) -> float:
        return max(self.membership_function)

    @property
    def center_of_mass(self) -> float:  # uncheck
        if self.height == 0:
            return math.floor((self.min_x + self.max_x)/2)

        numerator = 0
        denominator = 0
        for i, member_value in enumerate(self.membership_function):
            numerator += (self.min_x + i*(self.max_x-self.min_x)/self.number_of_points)*member_value
            denominator += member_value
        return numerator/denominator

    def alpha_cut(self, alpha: float) -> 'FuzzySet':
        new_membership_function = []
        for i, member_value in enumerate(self.membership_function):
            if member_value >= alpha:
                new_membership_function.append(alpha)
            else:
                new_membership_function.append(member_value)
        return FuzzySet(self.min_x, self.max_x, new_membership_function)

    def infer_membership(self, x: float) -> float:
        if x <= self.min_x:
            return self.membership_function[0]
        elif x >= self.max_x:
            return self.membership_function[-1]
        else:
            print("Fix", math.floor((x-self.min_x)/(self.max_x-self.min_x)*self.number_of_points)-1)
            return self.membership_function[math.floor((x-self.min_x)/(self.max_x-self.min_x)*self.number_of_points)]


class FuzzyRule():
    def __init__(self, antecedents: list[FuzzySet], consequent: FuzzySet) -> None:
        self.antecedents = antecedents
        self.consequent = consequent

    def infer(self, variable: tuple) -> FuzzySet:
        if len(variable) != len(self.antecedents):
            raise ValueError("The number of variables must be the same as the number of antecedents")

        # alpha = min([antecedent.infer_membership(x) for x, antecedent in zip(variable, self.antecedents)])
        alphas = []
        for x, antecedent in zip(variable, self.antecedents):
            print('x, set name:', x, antecedent.name)
            alphas.append(antecedent.infer_membership(x))
        
        min_alpha = min(alphas)
        print('alphas', alphas)
        print('min_alpha', min_alpha)


        return self.consequent.alpha_cut(min_alpha)

class FuzzySystem():
    def __init__(self, rules: list[FuzzyRule]) -> None:
        self.rules = rules

    def infer(self, input: tuple, infer_type: str = "avg_of_center") -> float:

        if infer_type == "avg_of_center":
            numerator = 0
            denominator = 0
            for i, rule in enumerate(self.rules):
                print(f"\n==== Rule {i} ===")
                result_set = rule.infer(input)
                assert -40 <= result_set.center_of_mass and result_set.center_of_mass <= 40, "center of mass must be positive"
                assert result_set.height >= 0, "height must be positive"

                print("in start of fuzzy system")
                print(f"center of mass: {result_set.center_of_mass}")
                print(f"height: {result_set.height}")
                numerator += result_set.center_of_mass * result_set.height
                denominator += result_set.height
            print("\n === in fuzzy system === ")
            print(f"{numerator}/{denominator}")

            angle = numerator/denominator

            assert -40 <= angle and angle <= 40, "angle should be within -40~40"

            return numerator/denominator
        
        else:
            raise NotImplementedError


MAX_CENTER_DISTANCE = 20
MIN_CENTER_DISTANCE = 0
MAX_COMBINE_DISTANCE = 10
MIN_COMBINE_DISTANCE = -10
TURN_LEFT = -40
TURN_RIGHT = 40
NUMBER_OF_POINTS_IN_FUZZY_SET = 1000

if __name__ == "__main__":
    MAX_X = -100
    MIN_X = 100
    mean = 0
    sigma = 100
    gaussian_func = lambda x, mean, sigma: np.exp(-0.5*((x-mean)/sigma)**2)
    triangular = lambda x, a, b, c: 0 if x <= a else (x-a)/(b-a) if a <= x <= b else (c-x)/(c-b) if b <= x <= c else 0
    unit_step = lambda x: 1 if x >= 0 else 0
    down_up = lambda x, a, b: 0 if x <= a else (x-a)/(b-a) if a <= x <= b else 1 
    up_down = lambda x, a, b: 1 if x <= a else (b-x)/(b-a) if a <= x <= b else 0 

    member_function = [up_down(x, -20, 0) for x in np.linspace(MIN_X, MAX_X, NUMBER_OF_POINTS_IN_FUZZY_SET)]
    member_function2 = [down_up(x, 0, 20) for x in np.linspace(MIN_X, MAX_X, NUMBER_OF_POINTS_IN_FUZZY_SET)]


    m_neg_set = FuzzySet(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, 
                    [up_down(x, 4, 5) for x in np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                    name="m_neg_set")
    m_mid_set = FuzzySet(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, 
                        [triangular(x, 4, 8, 12) for x in np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                        name="m_mid_set")
    m_pos_set = FuzzySet(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, 
                        [down_up(x, 11, 12) for x in np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                        name="m_pos_set")
    
    lr_neg_set = FuzzySet(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, 
                        [up_down(x, -15, 0) for x in np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                        name="lr_neg_set"
                        )
    lr_mid_set = FuzzySet(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, 
                        [triangular(x, -15, 0, 15) for x in np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                        name="lr_mid_set")
    lr_pos_set = FuzzySet(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, 
                        [down_up(x, 0, 15) for x in np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                        name="lr_pos_set")


    angle_neg_set = FuzzySet(TURN_LEFT, TURN_RIGHT, 
                            [up_down(x, -30, 0) for x in np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                            name="angle_neg_set")
    angle_mid_set = FuzzySet(TURN_LEFT, TURN_RIGHT, 
                            [triangular(x, -30, 0, 30) for x in np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                            name="angle_mid_set")
    angle_pos_set = FuzzySet(TURN_LEFT, TURN_RIGHT, 
                            [down_up(x, 0, 30) for x in np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET)],
                            name="angle_pos_set")

    fuzzy_rule_list = [
            FuzzyRule([m_neg_set, lr_neg_set], angle_pos_set),
            FuzzyRule([m_neg_set, lr_mid_set], angle_neg_set),
            FuzzyRule([m_neg_set, lr_pos_set], angle_neg_set),

            FuzzyRule([m_mid_set, lr_neg_set], angle_pos_set),
            FuzzyRule([m_mid_set, lr_mid_set], angle_mid_set),
            FuzzyRule([m_mid_set, lr_pos_set], angle_neg_set),

            FuzzyRule([m_pos_set, lr_neg_set], angle_mid_set),
            FuzzyRule([m_pos_set, lr_mid_set], angle_mid_set),
            FuzzyRule([m_pos_set, lr_pos_set], angle_mid_set)
        ]


    fuzzy_system = FuzzySystem(fuzzy_rule_list)

    fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    axs[0].plot(np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET), m_neg_set.membership_function, label=m_neg_set.name)
    axs[0].plot(np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET), m_mid_set.membership_function, label=m_mid_set.name)
    axs[0].plot(np.linspace(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET), m_pos_set.membership_function, label=m_pos_set.name)
    axs[0].legend()
    axs[0].set_xticks(np.arange(MIN_CENTER_DISTANCE, MAX_CENTER_DISTANCE, 1))
    axs[0].set_title("Center Distance")

    axs[1].plot(np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET), lr_neg_set.membership_function, label=lr_neg_set.name)
    axs[1].plot(np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET), lr_mid_set.membership_function, label=lr_mid_set.name)
    axs[1].plot(np.linspace(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, NUMBER_OF_POINTS_IN_FUZZY_SET), lr_pos_set.membership_function, label=lr_pos_set.name)
    axs[1].legend()
    axs[1].set_xticks(np.arange(MIN_COMBINE_DISTANCE, MAX_COMBINE_DISTANCE, 1))
    axs[1].set_title("Combine Distance")

    axs[2].plot(np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET), angle_neg_set.membership_function, label=angle_neg_set.name)
    axs[2].plot(np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET), angle_mid_set.membership_function, label=angle_mid_set.name)
    axs[2].plot(np.linspace(TURN_LEFT, TURN_RIGHT, NUMBER_OF_POINTS_IN_FUZZY_SET), angle_pos_set.membership_function, label=angle_pos_set.name)
    axs[2].legend()
    axs[2].set_xticks(np.arange(TURN_LEFT, TURN_RIGHT, 10))
    axs[2].set_title("Angle")
    plt.tight_layout()
    plt.show()
