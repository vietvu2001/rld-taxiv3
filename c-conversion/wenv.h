#ifndef WENV_H
#define WENV_H
#include <iostream>
#include <vector>
#include <algorithm>

int randint(int min, int max);

struct holder
{
    std::vector <int> s;
    int r;
    bool d;

    holder(std::vector <int> x, int y, bool z);
};

void display_vector(std::vector <int> a);


struct WindyGridworld
{
    // size of grid
    int width = 7;
    int length = 8;
    int max_row = width - 1;
    int max_col = length - 1;

    // wind strengths
    int wind[8] = {0, 0, 1, 1, 1, 0, 0, 0};

    // destinations
    std::vector <std::vector<int>> dest
    {
        {1, 6},
        {3, 5},
        {4, 7},
    };

    std::vector <int> current;
    std::vector <std::vector <int>> special;
    std::vector <std::vector <int>> jump_cells;

    bool terminal(std::vector <int> state);

    std::vector <int> reset();

    holder step(int action);

    std::vector <std::vector <int>> resettable_states();

    void render();
};

#endif