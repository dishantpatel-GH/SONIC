/**
 * @file inspire_hands.hpp
 * @brief Driver for Inspire robotic hands (left + right) on G1.
 *
 * InspireHands manages two 6-DOF Inspire hands via a single Unitree DDS topic pair
 * (MotorCmds_ / MotorStates_). Motor layout: indices 0-5 = right hand, 6-11 = left hand.
 * It provides:
 *  - Thread-safe command and state buffers (via DataBuffer).
 *  - writeOnce() – publishes combined 12-motor command with delta-q smoothing
 *    and close-ratio limiting.
 *  - Convenience helpers: open(), close(), hold(), stop().
 *  - setAllJointsCommand(is_left, q[7]) – uses first 6 elements; 7th ignored for API compatibility.
 *
 * ## Close-Ratio Limiting
 *
 * A runtime-adjustable max_close_ratio_ (range [0.2, 1.0]) limits how far the
 * fingers can close. 1.0 allows full closure; 0.2 keeps them mostly open.
 *
 * ## DDS Topics
 *
 *   Direction | Topic
 *   ----------|------------------
 *   Command   | rt/inspire/cmd   (MotorCmds_, 12 motors)
 *   State     | rt/inspire/state (MotorStates_, 12 motors)
 */

#ifndef INSPIRE_HANDS_HPP
#define INSPIRE_HANDS_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <iostream>

#include <unitree/idl/go2/MotorCmds_.hpp>
#include <unitree/idl/go2/MotorStates_.hpp>
#include <unitree/robot/channel/channel_factory.hpp>
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>

#include "utils.hpp"

static constexpr int INSPIRE_MOTOR_PER_HAND = 6;
static constexpr int INSPIRE_TOTAL_MOTORS = 12;  // 0-5 right, 6-11 left

/**
 * @brief Snapshot of one hand's state (7 elements for pipeline compatibility; 7th is padding).
 */
struct InspireHandStateSnapshot {
    std::array<double, 7> q = {};
    std::array<double, 7> dq = {};
    bool has_data = false;
};

/**
 * @class InspireHands
 * @brief Manages two Inspire hands (left + right) over a single DDS cmd/state topic pair.
 *
 * Does not run its own thread – the owning class calls writeOnce() at the
 * desired cadence from the command-writer thread.
 */
class InspireHands
{
public:
    InspireHands() = default;

    void initialize(const std::string& networkInterface)
    {
        if (!networkInterface.empty())
        {
            unitree::robot::ChannelFactory::Instance()->Init(0, networkInterface.c_str());
        }

        const std::string cmdTopic = "rt/inspire/cmd";
        const std::string stateTopic = "rt/inspire/state";

        unitree_go::msg::dds_::MotorCmds_ init_cmd;
        init_cmd.cmds().resize(INSPIRE_TOTAL_MOTORS);
        for (int i = 0; i < INSPIRE_TOTAL_MOTORS; ++i)
        {
            init_cmd.cmds()[i].mode(1);
            init_cmd.cmds()[i].q(0.0f);
            init_cmd.cmds()[i].dq(0.0f);
            init_cmd.cmds()[i].kp(0.0f);
            init_cmd.cmds()[i].kd(0.0f);
            init_cmd.cmds()[i].tau(0.0f);
        }
        cmd_buffer_.SetData(init_cmd);

        publisher_.reset(new unitree::robot::ChannelPublisher<unitree_go::msg::dds_::MotorCmds_>(cmdTopic));
        subscriber_.reset(new unitree::robot::ChannelSubscriber<unitree_go::msg::dds_::MotorStates_>(stateTopic));
        publisher_->InitChannel();
        subscriber_->InitChannel([this](const void* message) { this->onState(message); }, 1);
    }

    void SetMaxCloseRatio(double ratio)
    {
        max_close_ratio_ = std::max(0.2, std::min(1.0, ratio));
    }

    double GetMaxCloseRatio() const { return max_close_ratio_; }

    void writeOnce()
    {
        constexpr double MAX_DELTA_Q = 0.25;

        auto cmdPtr = cmd_buffer_.GetDataWithTime().data;
        auto statePtr = state_buffer_.GetDataWithTime().data;

        if (!publisher_ || !cmdPtr || static_cast<int>(cmdPtr->cmds().size()) != INSPIRE_TOTAL_MOTORS)
            return;

        unitree_go::msg::dds_::MotorCmds_ smoothedCmd;
        smoothedCmd.cmds().resize(INSPIRE_TOTAL_MOTORS);

        for (int hand = 0; hand < 2; ++hand)
        {
            bool is_left = (hand == 1);
            const int base = is_left ? 6 : 0;
            const auto& maxL = is_left ? MAX_LIMITS_LEFT : MAX_LIMITS_RIGHT;
            const auto& minL = is_left ? MIN_LIMITS_LEFT : MIN_LIMITS_RIGHT;

            for (int i = 0; i < INSPIRE_MOTOR_PER_HAND; ++i)
            {
                int idx = base + i;
                double desired_q = static_cast<double>(cmdPtr->cmds()[idx].q());
                desired_q = clipToMaxOpen(desired_q, maxL[i], minL[i], max_close_ratio_);

                if (statePtr && static_cast<int>(statePtr->states().size()) == INSPIRE_TOTAL_MOTORS)
                {
                    double current_q = static_cast<double>(statePtr->states()[idx].q());
                    double delta = desired_q - current_q;
                    double clamped_delta = std::max(-MAX_DELTA_Q, std::min(MAX_DELTA_Q, delta));
                    desired_q = current_q + clamped_delta;
                }

                smoothedCmd.cmds()[idx].mode(1);
                smoothedCmd.cmds()[idx].q(static_cast<float>(desired_q));
                smoothedCmd.cmds()[idx].dq(0.0f);
                smoothedCmd.cmds()[idx].kp(0.0f);
                smoothedCmd.cmds()[idx].kd(0.0f);
                smoothedCmd.cmds()[idx].tau(0.0f);
            }
        }

        publisher_->Write(smoothedCmd);
    }

    InspireHandStateSnapshot getState(bool is_left) const
    {
        InspireHandStateSnapshot snap;
        auto statePtr = state_buffer_.GetDataWithTime().data;
        if (!statePtr || static_cast<int>(statePtr->states().size()) != INSPIRE_TOTAL_MOTORS)
            return snap;

        const int base = is_left ? 6 : 0;
        for (int i = 0; i < INSPIRE_MOTOR_PER_HAND; ++i)
        {
            snap.q[i] = static_cast<double>(statePtr->states()[base + i].q());
            snap.dq[i] = static_cast<double>(statePtr->states()[base + i].dq());
        }
        snap.q[6] = 0.0;
        snap.dq[6] = 0.0;
        snap.has_data = true;
        return snap;
    }

    void setAllJointsCommand(bool is_left, const std::array<double, 7>& q,
                            std::optional<std::array<double, 7>> dq = std::nullopt,
                            std::optional<std::array<double, 7>> kp = std::nullopt,
                            std::optional<std::array<double, 7>> kd = std::nullopt,
                            std::optional<std::array<double, 7>> tau = std::nullopt)
    {
        auto currentPtr = cmd_buffer_.GetDataWithTime().data;
        unitree_go::msg::dds_::MotorCmds_ cmd = currentPtr ? *currentPtr : unitree_go::msg::dds_::MotorCmds_();
        if (!currentPtr || static_cast<int>(cmd.cmds().size()) != INSPIRE_TOTAL_MOTORS)
        {
            cmd.cmds().resize(INSPIRE_TOTAL_MOTORS);
            for (int i = 0; i < INSPIRE_TOTAL_MOTORS; ++i)
            {
                cmd.cmds()[i].mode(1);
                cmd.cmds()[i].q(0.0f);
                cmd.cmds()[i].dq(0.0f);
                cmd.cmds()[i].kp(0.0f);
                cmd.cmds()[i].kd(0.0f);
                cmd.cmds()[i].tau(0.0f);
            }
        }

        const int base = is_left ? 6 : 0;
        for (int i = 0; i < INSPIRE_MOTOR_PER_HAND; ++i)
        {
            auto& m = cmd.cmds()[base + i];
            m.mode(1);
            m.q(static_cast<float>(q[i]));
            m.dq(dq ? static_cast<float>((*dq)[i]) : 0.0f);
            m.kp(kp ? static_cast<float>((*kp)[i]) : 0.0f);
            m.kd(kd ? static_cast<float>((*kd)[i]) : 0.0f);
            m.tau(tau ? static_cast<float>((*tau)[i]) : 0.0f);
        }
        cmd_buffer_.SetData(std::move(cmd));
    }

    void stop(bool is_left)
    {
        auto currentPtr = cmd_buffer_.GetDataWithTime().data;
        unitree_go::msg::dds_::MotorCmds_ cmd = currentPtr ? *currentPtr : unitree_go::msg::dds_::MotorCmds_();
        if (cmd.cmds().size() != static_cast<size_t>(INSPIRE_TOTAL_MOTORS))
            cmd.cmds().resize(INSPIRE_TOTAL_MOTORS);
        const int base = is_left ? 6 : 0;
        for (int i = 0; i < INSPIRE_MOTOR_PER_HAND; ++i)
        {
            auto& m = cmd.cmds()[base + i];
            m.mode(1);
            m.q(0.0f);
            m.dq(0.0f);
            m.kp(0.0f);
            m.kd(0.0f);
            m.tau(0.0f);
        }
        cmd_buffer_.SetData(std::move(cmd));
    }

    void hold(bool is_left, double kp = 1.5, double kd = 0.1)
    {
        auto statePtr = state_buffer_.GetDataWithTime().data;
        if (!statePtr || static_cast<int>(statePtr->states().size()) != INSPIRE_TOTAL_MOTORS)
            return;
        const int base = is_left ? 6 : 0;
        unitree_go::msg::dds_::MotorCmds_ cmd;
        cmd.cmds().resize(INSPIRE_TOTAL_MOTORS);
        for (int i = 0; i < INSPIRE_TOTAL_MOTORS; ++i)
        {
            cmd.cmds()[i].mode(1);
            cmd.cmds()[i].q(0.0f);
            cmd.cmds()[i].dq(0.0f);
            cmd.cmds()[i].kp(0.0f);
            cmd.cmds()[i].kd(0.0f);
            cmd.cmds()[i].tau(0.0f);
        }
        for (int i = 0; i < INSPIRE_MOTOR_PER_HAND; ++i)
        {
            auto& m = cmd.cmds()[base + i];
            m.q(statePtr->states()[base + i].q());
            m.kp(static_cast<float>(kp));
            m.kd(static_cast<float>(kd));
        }
        cmd_buffer_.SetData(std::move(cmd));
    }

    void close(bool is_left, double kp = 1.5, double kd = 0.1)
    {
        const auto& maxL = is_left ? MAX_LIMITS_LEFT : MAX_LIMITS_RIGHT;
        const auto& minL = is_left ? MIN_LIMITS_LEFT : MIN_LIMITS_RIGHT;
        auto currentPtr = cmd_buffer_.GetDataWithTime().data;
        unitree_go::msg::dds_::MotorCmds_ cmd = currentPtr ? *currentPtr : unitree_go::msg::dds_::MotorCmds_();
        if (cmd.cmds().size() != static_cast<size_t>(INSPIRE_TOTAL_MOTORS))
            cmd.cmds().resize(INSPIRE_TOTAL_MOTORS);
        const int base = is_left ? 6 : 0;
        for (int i = 0; i < INSPIRE_MOTOR_PER_HAND; ++i)
        {
            double mid = (maxL[i] + minL[i]) / 2.0;
            auto& m = cmd.cmds()[base + i];
            m.mode(1);
            m.q(static_cast<float>(mid));
            m.dq(0.0f);
            m.kp(static_cast<float>(kp));
            m.kd(static_cast<float>(kd));
            m.tau(0.0f);
        }
        cmd_buffer_.SetData(std::move(cmd));
    }

    void open(bool is_left, double kp = 1.5, double kd = 0.1)
    {
        auto currentPtr = cmd_buffer_.GetDataWithTime().data;
        unitree_go::msg::dds_::MotorCmds_ cmd = currentPtr ? *currentPtr : unitree_go::msg::dds_::MotorCmds_();
        if (cmd.cmds().size() != static_cast<size_t>(INSPIRE_TOTAL_MOTORS))
            cmd.cmds().resize(INSPIRE_TOTAL_MOTORS);
        const int base = is_left ? 6 : 0;
        for (int i = 0; i < INSPIRE_MOTOR_PER_HAND; ++i)
        {
            auto& m = cmd.cmds()[base + i];
            m.mode(1);
            m.q(0.0f);
            m.dq(0.0f);
            m.kp(static_cast<float>(kp));
            m.kd(static_cast<float>(kd));
            m.tau(0.0f);
        }
        cmd_buffer_.SetData(std::move(cmd));
    }

private:
    using MotorCmds = unitree_go::msg::dds_::MotorCmds_;
    using MotorStates = unitree_go::msg::dds_::MotorStates_;
    using ChannelPublisherPtr = unitree::robot::ChannelPublisherPtr<MotorCmds>;
    using ChannelSubscriberPtr = unitree::robot::ChannelSubscriberPtr<MotorStates>;

    ChannelPublisherPtr publisher_;
    ChannelSubscriberPtr subscriber_;
    DataBuffer<MotorCmds> cmd_buffer_;
    DataBuffer<MotorStates> state_buffer_;
    double max_close_ratio_ = 1.0;

    void onState(const void* message)
    {
        const auto* incoming = static_cast<const MotorStates*>(message);
        state_buffer_.SetData(*incoming);
    }

    static double clipToMaxOpen(double desired_q, double max_limit, double min_limit, double max_close_ratio)
    {
        double q_max_open_pos = max_close_ratio * max_limit;
        double q_max_open_neg = max_close_ratio * min_limit;
        if (desired_q > 0.0 && max_limit > 0.0 && desired_q > q_max_open_pos)
            return q_max_open_pos;
        if (desired_q < 0.0 && min_limit < 0.0 && desired_q < q_max_open_neg)
            return q_max_open_neg;
        return desired_q;
    }

    // Inspire hand DFQ joint limits (per hand): thumb_yaw, thumb_pitch, index, middle, ring, pinky
    static constexpr std::array<double, INSPIRE_MOTOR_PER_HAND> MAX_LIMITS_LEFT  = { 1.3,  0.6,  1.7,  1.7,  1.7,  1.7 };
    static constexpr std::array<double, INSPIRE_MOTOR_PER_HAND> MIN_LIMITS_LEFT  = {-0.1, -0.1,  0.0,  0.0,  0.0,  0.0 };
    static constexpr std::array<double, INSPIRE_MOTOR_PER_HAND> MAX_LIMITS_RIGHT = { 1.3,  0.6,  1.7,  1.7,  1.7,  1.7 };
    static constexpr std::array<double, INSPIRE_MOTOR_PER_HAND> MIN_LIMITS_RIGHT = {-0.1, -0.1,  0.0,  0.0,  0.0,  0.0 };
};

#endif // INSPIRE_HANDS_HPP
