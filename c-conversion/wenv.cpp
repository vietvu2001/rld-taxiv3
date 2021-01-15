#include <iostream>
#include <vector>
#include <assert.h>
#include <time.h>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <string.h>
#include <experimental/random>
#include "wenv.h"
using namespace std;

int randint(int min, int max)
{
    int seed = rand();
    srand(seed);
    int x = experimental::randint(min, max);
    return x;
}

holder::holder(vector <int> x, float y, bool z)
{
    s = x;
    r = y;
    d = z;
}

void display_vector(vector<int> a)
{
    if (a.size() == 0)
    {
        cout << "None" << endl;
        return;
    }

    for (int i = 0; i < a.size(); i++)
    {
        cout << a[i] << " ";
    }
    cout << endl;
}

bool WindyGridworld::terminal(vector <int> state)
{
    assert(state.size() == 4);
    if (state[0] == dest[state[3]][0] && state[1] == dest[state[3]][1] && state[2] == 3) return true;
    return false;
}

vector <int> WindyGridworld::reset()
{
    int row = randint(0, max_row);
    int col = randint(0, 1);

    vector <int> goal_indices{0, 1, 2};
    random_shuffle(goal_indices.begin(), goal_indices.end());

    int ind_1 = goal_indices[0];
    int ind_2 = goal_indices[1];

    vector <int> state{row, col, ind_1, ind_2};

    current = state;

    return current;
}

holder WindyGridworld::step(int action)
{
    assert(current.size() == 4);
    assert(action >= 0 && action <= 11);

    int i = current[0];
    int j = current[1];

    assert(0 <= j && j <= 7);

    vector <int> pos{i, j};
    int next_row = i;
    int next_col = j;

    // Present reward
    float reward = -1.0;

    // Preprocess conditionals
    bool special_cell = false;
    bool jump_cell = false;

    int a = special.size();
    int b = jump_cells.size();

    if (a != 0)
    {
        for (int ind = 0; ind < a; ind++)
        {
            if (pos == special[ind])
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
            if (pos == jump_cells[ind])
            {
                jump_cell = true;
                break;
            }
        }
    }

    if (action == 0)
    {
        next_row = max(min(i + 1 - wind[j], max_row), 0);
        next_col = j;
    }

    else if (action == 1)
    {
        next_row = max(i - 1 - wind[j], 0);
        next_col = j;
    }

    else if (action == 2)
    {
        next_row = max(i - wind[j], 0);
        next_col = min(j + 1, max_col);
    }

    else if (action == 3)
    {
        next_row = max(i - wind[j], 0);
        next_col = max(j - 1, 0);
    }

    // Novel actions:
    // 4: move northeast
    // 5: move southeast
    // 6: move southwest
    // 7: move northwest

    else if (action == 4 && special_cell)
    {
        next_row = max(i - 1 - wind[j], 0);
        next_col = min(j + 1, max_col);
    }

    else if (action == 5 && special_cell)
    {
        next_row = max(min(i + 1 - wind[j], max_row), 0);
        next_col = min(j + 1, max_col);
    }

    else if (action == 6 && special_cell)
    {
        next_row = max(min(i + 1 - wind[j], max_row), 0);
        next_col = max(j - 1, 0);
    }

    else if (action == 7 && special_cell)
    {
        next_row = min(i - 1 - wind[j], 0);
        next_col = max(j - 1, 0);
    }

    // Novel actions (continued)
    // 8: jump south 2 steps
    // 9: jump north 2 steps
    // 10: jump east 2 steps
    // 11: jump west 2 steps

    else if (action == 8 && jump_cell)
    {
        next_row = max(min(i + 2 - wind[j], max_row), 0);
        next_col = j;
    }

    else if (action == 9 && jump_cell)
    {
        next_row = max(i - 2 - wind[j], 0);
        next_col = j;
    }

    else if (action == 10 && jump_cell)
    {
        next_row = max(i - wind[j], 0);
        next_col = min(j + 2, max_col);
    }

    else if (action == 11 && jump_cell)
    {
        next_row = max(i - wind[j], 0);
        next_col = max(j - 2, 0);
    }

    // Build vector of next state and change f_addr if possible
    vector <int> next_state{next_row, next_col, current[2], current[3]};

    if (next_state[2] != 3)
    {
        if (next_row == dest[current[2]][0] && next_col == dest[current[2]][1])
        {
            next_state[2] = 3;
        }
    }

    current = next_state;
    bool done = terminal(next_state);

    if (done) reward = 20.0;

    holder h(next_state, reward, done);

    return h;
}

vector <vector <int>> WindyGridworld::resettable_states()
{
    vector <vector <int>> ls;
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < 2; j++)
        {
            for (int ind_1 = 0; ind_1 < 3; ind_1++)
            {
                for (int ind_2 = 0; ind_2 < 3; ind_2++)
                {
                    if (ind_1 != ind_2)
                    {
                        vector <int> v{i, j, ind_1, ind_2};
                        ls.push_back(v);
                    }
                }
            }
        }
    }

    return ls;
}

void WindyGridworld::render()
{
    string outboard = "";
    int shape[2] = {width, length};

    // Make sure the environment is properly reset
    assert(current.size() == 4);

    for (int y = -1; y < width + 1; y++)
    {
        string outline = "";
        string output = "";

        for (int x = -1; x < length + 1; x++)
        {
            bool flag = false;
            for (int i = 0; i < dest.size(); i++)
            {
                if (y == dest[i][0] && x == dest[i][1])
                {
                    flag = true;
                    break;
                }
            }

            if (current[0] == y && current[1] == x)
            {
                output = "X";
            }

            else if (current[2] != 3 && y == dest[current[2]][0] && x == dest[current[2]][1])
            {
                output = "F";
            }

            else if (y == dest[current[3]][0] && x == dest[current[3]][1])
            {
                output = "S";
            }

            else if (flag)
            {
                output = "G";
            }

            else if (x == -1 or x == length or y == -1 or y == width)
            {
                output = "#";
            }

            else
            {
                output = " ";
            }

            if (x == shape[1])
            {
                output += "\n";
            }

            outline += output;
        }

        outboard += outline;
    }

    outboard += "\n";
    cout << outboard << endl;
    return;
}