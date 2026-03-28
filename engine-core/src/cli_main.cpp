#include "gomoku/engine.hpp"
#include "gomoku/types.hpp"

#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

using json = nlohmann::json;

namespace {

gomoku::Position parse_position(const json& j) {
    gomoku::Position pos;
    pos.board_size = j.value("boardSize", 15);
    pos.win_length = j.value("winLength", 5);

    if (j.contains("cells")) {
        auto& cells = j["cells"];
        for (int i = 0; i < static_cast<int>(cells.size()) && i < gomoku::MAX_CELLS; i++) {
            pos.cells[i] = static_cast<gomoku::Cell>(cells[i].get<int>());
        }
    }

    pos.side_to_move = static_cast<gomoku::Cell>(j.value("sideToMove", 1));
    pos.move_count = j.value("moveCount", 0);
    pos.last_move = j.value("lastMove", -1);

    if (j.contains("moveHistory")) {
        for (auto& mv : j["moveHistory"]) {
            pos.move_history.push_back(mv.get<int>());
        }
    }

    return pos;
}

json move_candidate_to_json(const gomoku::MoveCandidate& mc) {
    return json{
        {"move", mc.move},
        {"score", mc.score},
        {"row", mc.row},
        {"col", mc.col}
    };
}

json engine_result_to_json(const gomoku::EngineResult& r) {
    json j;
    j["bestMove"] = r.best_move;
    j["value"] = r.value;
    j["source"] = r.source;
    j["depth"] = r.depth;
    j["nodesSearched"] = r.nodes_searched;
    j["timeMs"] = r.time_ms;

    j["topMoves"] = json::array();
    for (auto& mc : r.top_moves) {
        j["topMoves"].push_back(move_candidate_to_json(mc));
    }

    j["pvLine"] = r.pv_line;
    j["policy"] = r.policy;

    return j;
}

} // anonymous namespace

int main() {
    gomoku::GomokuEngine engine;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        json request;
        try {
            request = json::parse(line);
        } catch (const json::parse_error& e) {
            json err;
            err["error"] = std::string("JSON parse error: ") + e.what();
            std::cout << err.dump() << "\n";
            continue;
        }

        std::string command = request.value("command", "");
        json response;

        if (command == "best-move") {
            auto pos = parse_position(request["position"]);
            auto result = engine.best_move(pos);
            response = engine_result_to_json(result);

        } else if (command == "analyze") {
            auto pos = parse_position(request["position"]);
            auto result = engine.best_move(pos);
            response = engine_result_to_json(result);
            response["evaluation"] = engine.evaluate(pos);

        } else if (command == "suggest") {
            auto pos = parse_position(request["position"]);
            int top_n = request.value("topN", 5);
            auto hints = engine.get_hints(pos, top_n);
            response["hints"] = json::array();
            for (auto& mc : hints) {
                response["hints"].push_back(move_candidate_to_json(mc));
            }

        } else if (command == "info") {
            response["engine"] = "gomoku-engine";
            response["version"] = "1.0.0";
            response["status"] = "stub";
            response["stats"] = engine.stats();

        } else {
            response["error"] = "Unknown command: " + command;
        }

        std::cout << response.dump() << "\n";
        std::cout.flush();
    }

    return 0;
}
