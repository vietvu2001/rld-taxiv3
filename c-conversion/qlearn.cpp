#include <iostream>
#include <vector>
#include <time.h>
#include <random>
#include <algorithm>
#include <bits/stdc++.h>
#include <numeric>
#include <utility>
#include <typeinfo>

#include "wenv.h"
using namespace std;

int choice(vector <int> v, vector <double> probs)
{
    assert(v.size() == probs.size());

    // Random number
    double val = (double)rand() / RAND_MAX;

    // Partial sums (cumulative sums)
    double result[probs.size()];

    partial_sum(probs.begin(), probs.end(), result);

    int a = -1;

    if (val < result[0]) a = v[0];

    else  // should have been binary search, but not due to developer's laziness
    {
        for (int i = 0; i < probs.size() - 1; i++)
        {
            if (result[i] <= val && val < result[i + 1])
            {
                a = v[i + 1];
                break;
            }
        }
    }

    return a;
}

struct w_QAgent
{
    WindyGridworld env;
    double alpha = 0.25;
    double epsilon = 0.1;
    double rate = 1.0;

    map <vector<int>, map <int, double>> q;

    vector <int> actions(vector <int> state)
    {
        vector <int> pos{state[0], state[1]};
        bool special_cell = false;
        bool jump_cell = false;

        int a = env.special.size();
        int b = env.jump_cells.size();

        if (a != 0)
        {
            for (int ind = 0; ind < a; ind++)
            {
                if (pos == env.special[ind])
                {
                    special_cell = true;
                    break;
                }
            }
        }

        if (b != 0)
        {
            for (int ind = 0; ind < b; ind++)
            {
                if (pos == env.jump_cells[ind])
                {
                    jump_cell = true;
                    break;
                }
            }
        }

        vector <int> actions;

        if (special_cell && jump_cell)
        {
            for (int ind = 0; ind < 12; ind++)
            {
                actions.push_back(ind);
            }
        }

        else if (special_cell)
        {
            for (int ind = 0; ind < 8; ind++)
            {
                actions.push_back(ind);
            }
        }

        else if (jump_cell)
        {
            for (int ind = 0; ind < 4; ind++)
            {
                actions.push_back(ind);
            }

            for (int ind = 8; ind < 12; ind++)
            {
                actions.push_back(ind);
            }
        }

        else
        {
            for (int ind = 0; ind < 4; ind++)
            {
                actions.push_back(ind);
            }
        }

        return actions;
    }

    double get_q_value(vector <int> state, int action)
    {
        map <vector<int>, map <int, double>>::iterator it;
        it = q.find(state);

        if (it == q.end()) return 0.0;

        else
        {
            map <int, double>::iterator it_1;
            it_1 = q[state].find(action);

            if (it_1 == q[state].end()) return 0.0;

            else return q[state][action];
        }
    }

    double best_reward_state(vector <int> state)
    {
        map <vector<int>, map <int, double>>::iterator it;
        it = q.find(state);

        if (it == q.end()) return 0.0;

        else
        {
            double opt_val = q[state].begin()->second;

            for (auto it_1 = q[state].begin(); it_1 != q[state].end(); it_1++)
            {
                double x = it_1->second;
                if (x > opt_val)
                {
                    opt_val = x;
                }
            }

            if (q[state].size() == actions(state).size()) return opt_val;

            else return max(opt_val, 0.0);
        }
    }

    int choose_action(vector <int> state, bool prob=true)
    {
        vector <int> vals{0, 1};
        vector <double> probs{epsilon, 1 - epsilon};

        int r = choice(vals, probs);
        vector <int> ls = actions(state);

        if (!prob || r == 1)
        {
            double opt_val = (double)INT_MIN;
            int action = -1;
            int N = ls.size();

            for (int i = 0; i < N; i++)
            {
                double value = get_q_value(state, ls[i]);
                if (value > opt_val)
                {
                    opt_val = value;
                    action = ls[i];
                }
            }

            assert(action != -1);
            return action;
        }

        else
        {
            int ind = randint(0, ls.size() - 1);
            return ls[ind];
        }
    }

    void qlearn(int num_episodes, bool show=true, int number=-1, bool render=true)
    {
        for (int i = 0; i < num_episodes; i++)
        {
            if (show && (i % 100 == 0 || i == num_episodes - 1))
            {
                if (number == -1) printf("Episode %d begins!\n", i);

                else printf("Episode %d begins! (%d)\n", i, number);
            }

            vector <int> s = env.reset();
            int t = 0;

            while (t < 2000)
            {
                int action = choose_action(s);
                holder h = env.step(action);

                vector <int> s_next = h.s;
                double reward = (double)h.r;
                bool done = h.d;

                double future_rewards_estimated = best_reward_state(s_next);
                double old_q = get_q_value(s, action);

                // Update q-value
                q[s][action] = old_q + alpha * (reward + rate * future_rewards_estimated - old_q);

                s = s_next;

                if (done) break;

                t += 1;
            }

            if (show && (i % 100 == 0 || i == num_episodes - 1))
            {
                if (number == -1) printf("Episode %d done! (%d)\n", i, t);

                else printf("Episode %d begins! (%d) (%d)\n", i, t, number);

                if (render) env.render();
            }
        }
    }

    pair <double, vector <vector <int>>> eval(bool display, vector <int> fixed)
    {
        vector <int> s = env.reset();

        // Use fixed vector
        env.current = fixed;
        s = env.current;

        vector <vector <int>> states{s};

        int t = 0;
        double total = 0.0;

        while (t < 1000)
        {
            int action = choose_action(s, false);
            holder h = env.step(action);

            vector <int> s_next = h.s;
            double reward = (double)h.r;
            bool done = h.d;

            states.push_back(s_next);
            total += reward;

            s = s_next;
            t += 1;

            if (done) break;
        }

        if (display) env.render();

        return make_pair(total, states);
    }
};

int main()
{
    srand(time(NULL));
    w_QAgent agent;

    clock_t start = clock();
    agent.qlearn(3000, true, -1, false);
    clock_t end = clock();

    vector <vector <int>> ls = agent.env.resettable_states();
    vector <double> values;

    for (int i = 0; i < ls.size(); i++)
    {
        double x = agent.eval(false, ls[i]).first;
        values.push_back(x);
    }

    cout << "Time taken: " << (float)(end - start) / CLOCKS_PER_SEC << endl;

    double sum = 0.0;
    for (int i = 0; i < values.size(); i++)
    {
        sum += values[i];
    }

    cout << sum / values.size() << endl;
    cout << values.size() << endl;

    return 0;
}