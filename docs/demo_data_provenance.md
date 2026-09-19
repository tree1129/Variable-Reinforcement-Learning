# 仿真 Demo 和数据截图来源

## 仿真回放

- 来源 checkpoint：`case4_7_sac_dual_arm_stable_grasp_v9_4gpu/gpu_6/sac_case4_7_best.zip`
- 回放范围：Case 4、Case 5、Case 6、Case 7
- 视频格式：GIF 和 MP4
- 每个回放显示抓取、插入和回撤阶段

## 数据图表

`assets/data/case_success_and_steps.png` 和 `assets/data/safety_quality_summary.png` 根据 `demo/evaluation/stable_evaluation_20.json` 生成。统计字段包括：

- `success`
- `score`
- `safety`
- `steps`
- `max_grasp_jump_m`
- `gripper_open_while_grasped_frames`

本批次为固定场景回归，样本数为 20；不应外推为随机场景或真机成功率。
