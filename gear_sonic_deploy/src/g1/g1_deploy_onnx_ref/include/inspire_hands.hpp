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
 * ## Motor Layout (per hand, per Unitree docs)
 *
 *   DDS Index | Joint
 *   ----------|------------------
 *   0 / 6     | pinky
 *   1 / 7     | ring
 *   2 / 8     | middle
 *   3 / 9     | index
 *   4 / 10    | thumb bend (pitch)
 *   5 / 11    | thumb rotation (yaw)
 *
 * ## Pipeline Joint Order (from Python solver)
 *
 *   Pipeline Index | Joint
 *   ---------------|------------------
 *   0              | thumb_yaw     [-0.1, 1.3] rad
 *   1              | thumb_pitch   [-0.1, 0.6] rad
 *   2              | index         [ 0.0, 1.7] rad
 *   3              | middle        [ 0.0, 1.7] rad
 *   4              | ring          [ 0.0, 1.7] rad
 *   5              | pinky         [ 0.0, 1.7] rad
 *   6              | padding       (always 0)
 *
 * ## Hardware Command Format
 *
 * The Inspire DFX hardware expects q values in [0.0, 1.0]:
 *   - 0.0 = fully closed
 *   - 1.0 = fully open
 *
 * Conversion: hw_q = clamp((max_rad - pipeline_rad) / (max_rad - min_rad), 0, 1)
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
            init_cmd.cmds()[i].q(1.0f);  // 1.0 = fully open in hardware convention
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
        // Max change per cycle in [0, 1] normalized space.
        // Smoothing is against last_sent_q_ (our own previous command), NOT against
        // DDS state feedback, because the state topic returns raw encoder values
        // (not [0,1] normalized), which would corrupt the smoothing calculation.
        constexpr double MAX_DELTA_Q = 0.1;

        auto cmdPtr = cmd_buffer_.GetDataWithTime().data;

        if (!publisher_ || !cmdPtr || static_cast<int>(cmdPtr->cmds().size()) != INSPIRE_TOTAL_MOTORS)
            return;

        unitree_go::msg::dds_::MotorCmds_ smoothedCmd;
        smoothedCmd.cmds().resize(INSPIRE_TOTAL_MOTORS);

        for (int i = 0; i < INSPIRE_TOTAL_MOTORS; ++i)
        {
            double desired_q = static_cast<double>(cmdPtr->cmds()[i].q());

            // Clamp to [0, 1] range
            desired_q = std::max(0.0, std::min(1.0, desired_q));

            // Apply max_close_ratio: prevent closing below this threshold
            // In hw convention: 0=closed, 1=open. Limit minimum to (1 - max_close_ratio)
            double min_allowed = 1.0 - max_close_ratio_;
            if (desired_q < min_allowed)
                desired_q = min_allowed;

            // Delta-q smoothing against last sent command
            double delta = desired_q - last_sent_q_[i];
            double clamped_delta = std::max(-MAX_DELTA_Q, std::min(MAX_DELTA_Q, delta));
            desired_q = last_sent_q_[i] + clamped_delta;

            smoothedCmd.cmds()[i].mode(1);
            smoothedCmd.cmds()[i].q(static_cast<float>(desired_q));
            smoothedCmd.cmds()[i].dq(0.0f);
            smoothedCmd.cmds()[i].kp(0.0f);
            smoothedCmd.cmds()[i].kd(0.0f);
            smoothedCmd.cmds()[i].tau(0.0f);

            last_sent_q_[i] = desired_q;
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

        // Read hardware state (in [0,1] hw motor order) and convert to pipeline format
        // Hardware order: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        // Pipeline order: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
        for (int pi = 0; pi < INSPIRE_MOTOR_PER_HAND; ++pi)
        {
            int hw_motor = PIPELINE_TO_HW[pi];
            double hw_q = static_cast<double>(statePtr->states()[base + hw_motor].q());
            double hw_dq = static_cast<double>(statePtr->states()[base + hw_motor].dq());

            // Denormalize from [0, 1] hardware convention to radians in pipeline convention
            // hw_q: 0=closed, 1=open.  pipeline: 0=open, max=closed
            // rad = max - hw_q * (max - min)
            double range = PIPE_MAX_LIMITS[pi] - PIPE_MIN_LIMITS[pi];
            snap.q[pi] = PIPE_MAX_LIMITS[pi] - hw_q * range;
            snap.dq[pi] = -hw_dq * range;  // negate because hw and pipeline have opposite sign
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
                cmd.cmds()[i].q(1.0f);  // default to open
                cmd.cmds()[i].dq(0.0f);
                cmd.cmds()[i].kp(0.0f);
                cmd.cmds()[i].kd(0.0f);
                cmd.cmds()[i].tau(0.0f);
            }
        }

        const int base = is_left ? 6 : 0;

        // Convert from pipeline format (radians) to hardware format ([0,1] normalized)
        // and reorder from pipeline order to hardware motor order.
        //
        // Pipeline: [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
        // Hardware: [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
        //
        // Normalization: hw_q = clamp((max_rad - val_rad) / (max_rad - min_rad), 0, 1)
        //   This maps: val=0 (open) → hw_q=1 (open), val=max (closed) → hw_q=0 (closed)
        for (int pi = 0; pi < INSPIRE_MOTOR_PER_HAND; ++pi)
        {
            int hw_motor = PIPELINE_TO_HW[pi];
            double range = PIPE_MAX_LIMITS[pi] - PIPE_MIN_LIMITS[pi];
            double hw_q = (range > 1e-6)
                ? std::max(0.0, std::min(1.0, (PIPE_MAX_LIMITS[pi] - q[pi]) / range))
                : 0.5;

            auto& m = cmd.cmds()[base + hw_motor];
            m.mode(1);
            m.q(static_cast<float>(hw_q));
            m.dq(0.0f);
            m.kp(0.0f);
            m.kd(0.0f);
            m.tau(0.0f);
        }
        cmd_buffer_.SetData(std::move(cmd));
    }

    void stop(bool is_left)
    {
        // Open hands (q=0 in pipeline = open) with no holding torque
        std::array<double, 7> q_sim = {0.0};
        setAllJointsCommand(is_left, q_sim);
    }

    void hold(bool is_left, double kp = 1.5, double kd = 0.1)
    {
        auto statePtr = state_buffer_.GetDataWithTime().data;
        if (!statePtr || static_cast<int>(statePtr->states().size()) != INSPIRE_TOTAL_MOTORS)
            return;
        const int base = is_left ? 6 : 0;
        // hold() writes raw hardware q back directly — no conversion needed
        // since we read hardware state and write it back to hardware
        unitree_go::msg::dds_::MotorCmds_ cmd;
        cmd.cmds().resize(INSPIRE_TOTAL_MOTORS);
        for (int i = 0; i < INSPIRE_TOTAL_MOTORS; ++i)
        {
            cmd.cmds()[i].mode(1);
            cmd.cmds()[i].q(1.0f);
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
        // Close = mid-range in pipeline convention
        std::array<double, 7> q_sim = {0.0};
        std::array<double, 7> kp_arr = {0.0};
        std::array<double, 7> kd_arr = {0.0};
        for (int i = 0; i < INSPIRE_MOTOR_PER_HAND; ++i)
        {
            q_sim[i] = (PIPE_MAX_LIMITS[i] + PIPE_MIN_LIMITS[i]) / 2.0;
            kp_arr[i] = kp;
            kd_arr[i] = kd;
        }
        setAllJointsCommand(is_left, q_sim, std::nullopt, kp_arr, kd_arr);
    }

    void open(bool is_left, double kp = 1.5, double kd = 0.1)
    {
        // Open = 0 in pipeline convention
        std::array<double, 7> q_sim = {0.0};
        std::array<double, 7> kp_arr = {0.0};
        std::array<double, 7> kd_arr = {0.0};
        for (int i = 0; i < INSPIRE_MOTOR_PER_HAND; ++i)
        {
            kp_arr[i] = kp;
            kd_arr[i] = kd;
        }
        setAllJointsCommand(is_left, q_sim, std::nullopt, kp_arr, kd_arr);
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
    std::array<double, INSPIRE_TOTAL_MOTORS> last_sent_q_ = {
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0,   // right hand: all open
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0};  // left hand: all open

    void onState(const void* message)
    {
        const auto* incoming = static_cast<const MotorStates*>(message);
        state_buffer_.SetData(*incoming);
    }

    // ---- Joint limit and reorder tables ----

    // Pipeline joint limits (in pipeline order: thumb_yaw, thumb_pitch, index, middle, ring, pinky)
    static constexpr std::array<double, INSPIRE_MOTOR_PER_HAND> PIPE_MAX_LIMITS = { 1.3,  0.6,  1.7,  1.7,  1.7,  1.7 };
    static constexpr std::array<double, INSPIRE_MOTOR_PER_HAND> PIPE_MIN_LIMITS = {-0.1, -0.1,  0.0,  0.0,  0.0,  0.0 };

    // Mapping from pipeline index → hardware DDS motor index (within a single hand)
    //
    // Pipeline: [0:thumb_yaw, 1:thumb_pitch, 2:index, 3:middle, 4:ring, 5:pinky]
    // Hardware: [0:pinky,     1:ring,        2:middle, 3:index,  4:thumb_pitch, 5:thumb_yaw]
    static constexpr std::array<int, INSPIRE_MOTOR_PER_HAND> PIPELINE_TO_HW = {5, 4, 3, 2, 1, 0};
};

#endif // INSPIRE_HANDS_HPP
