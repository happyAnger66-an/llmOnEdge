/*
 * SPDX-License-Identifier: Apache-2.0
 */

#include "llm_on_edge/engine/engine_runner.h"

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

using llm_on_edge::engine::EngineRunner;

TEST(EngineRunner, LoadConfigText)
{
    char path[] = "/tmp/edge_llm_engine_cfg_XXXXXX.json";
    int const fd = mkstemps(path, 5);
    ASSERT_NE(fd, -1);
    (void)close(fd);

    {
        std::ofstream f(path);
        ASSERT_TRUE(static_cast<bool>(f));
        f << R"({"model":"demo","max_batch_size":4})";
    }

    auto txt = EngineRunner::load_config_text(path);
    EXPECT_NE(txt.find("\"model\":\"demo\""), std::string::npos);
    EXPECT_NE(txt.find("\"max_batch_size\":4"), std::string::npos);

    std::remove(path);
}

