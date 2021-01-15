#include <iostream>
#include <vector>
#include <time.h>
#include <random>
#include <algorithm>
#include <bits/stdc++.h>
#include <numeric>
#include <utility>
#include <unistd.h>
#include <cstdlib>

#include "wenv.h"
using namespace std;

int choice(vector <int> v, vector <float> probs)
{
    assert(v.size() == probs.size());

    // Random number
    float val = (float)rand() / RAND_MAX;

    // Partial sums (cumulative sums)
    float result[probs.size()];

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
    float alpha = 0.5;
    float epsilon = 1;
    float rate = 1.0;

    map <vector<int>, map <int, float>> q;

    w_QAgent(WindyGridworld e)
    {
        env = e;
    }

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

    float get_q_value(vector <int> state, int action)
    {
        map <vector<int>, map <int, float>>::iterator it;
        it = q.find(state);

        if (it == q.end()) return 0.0;

        else
        {
            map <int, float>::iterator it_1;
            it_1 = q[state].find(action);

            if (it_1 == q[state].end()) return 0.0;

            else return q[state][action];
        }
    }

    float best_reward_state(vector <int> state)
    {
        map <vector<int>, map <int, float>>::iterator it;
        it = q.find(state);

        if (it == q.end()) return 0.0;

        else
        {
            float opt_val = q[state].begin()->second;

            for (auto it_1 = q[state].begin(); it_1 != q[state].end(); it_1++)
            {
                float x = it_1->second;
                if (x > opt_val)
                {
                    opt_val = x;
                }
            }

            if (q[state].size() == actions(state).size()) return opt_val;

            else return max(opt_val, (float)0);
        }
    }

    int choose_action(vector <int> state, bool prob=true)
    {
        vector <int> vals{0, 1};
        vector <float> probs{epsilon, 1 - epsilon};

        int r = choice(vals, probs);
        vector <int> ls = actions(state);

        if (!prob || r == 1)
        {
            float opt_val = (float)INT_MIN;
            int action = -1;
            int N = ls.size();

            for (int i = 0; i < N; i++)
            {
                float value = get_q_value(state, ls[i]);
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
            // srand(i);
            if (show && (i % 100 == 0 || i == num_episodes - 1))
            {
                if (number == -1) printf("Episode %d begins!\n", i);

                else printf("Episode %d begins! (%d)\n", i, number);
            }

            vector <int> s = env.reset();
            int t = 0;

            while (t < 2000)
            {
                if (t % 100 == 0) srand(2000 * i + t);
                int action = choose_action(s);
                holder h = env.step(action);

                vector <int> s_next = h.s;
                float reward = h.r;
                bool done = h.d;

                float future_rewards_estimated = best_reward_state(s_next);
                float old_q = get_q_value(s, action);

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

        if (epsilon > 0.05) epsilon *= 0.995;
    }

    pair <float, vector <vector <int>>> eval(bool display, vector <int> fixed)
    {
        vector <int> s = env.reset();

        // Use fixed vector
        env.current = fixed;
        s = env.current;

        vector <vector <int>> states{s};

        int t = 0;
        float total = 0.0;

        while (t < 1000)
        {
            int action = choose_action(s, false);
            holder h = env.step(action);

            vector <int> s_next = h.s;
            float reward = h.r;
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
    WindyGridworld env;

    // Modify environment
    vector <int> p{1, 5};
    env.jump_cells.push_back(p);
    w_QAgent agent(env);

    clock_t start = clock();
    agent.qlearn(3000, true, -1, false);
    clock_t end = clock();

    vector <vector <int>> ls = agent.env.resettable_states();
    vector <float> values;

    for (int i = 0; i < ls.size(); i++)
    {
        float x = agent.eval(false, ls[i]).first;
        values.push_back(x);
    }

    cout << "Time taken: " << (float)(end - start) / CLOCKS_PER_SEC << endl;

    float sum = 0.0;
    for (int i = 0; i < values.size(); i++)
    {
        sum += values[i];
    }

    cout << (float)sum / values.size() << endl;
    agent.env.render();

    return 0;
}