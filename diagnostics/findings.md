# 去噪检查证据

- 当前工作区原有修改：Practical_Project.ipynb、reverse_process_student.png/gif；pokemon_ddpm_student.pt 为原有未跟踪文件。
- Notebook cells 29/31/37/40 的 schedule、q_sample、epsilon MSE、reverse mean/posterior std 静态检查未发现公式错误。
- NUM_DIFFUSION_STEPS=1000，alpha_bar 最后值约 4.0358e-5。
- 训练记录 100 epochs，loss 0.673173 -> 0.049650。
- tensor_to_img 将 [-1,1] 映射到 [0,1] 后截断；需要检查截断前数值。
- sample_images 使用当前内存 model，未显式加载 checkpoint；是否被重建需用保存权重验证。
- notebook 保存的早期输出提示 PyTorch 不支持 RTX 5060 的 sm_120；诊断优先 CPU。

## CPU 实验
- 严格加载现有 checkpoint，通过已知真实噪声的 forward/oracle reverse 数学测试（float64 排除抵消误差）。
- CPU/float32，seed42，batch4，1000步：原最终范围[-3.7656,2.0295]，24.72%通道值<-1，6.49%>1，RGB均值[0.4625,-0.2562,-0.9428]。
- 同噪声下仅替换为 x0_hat 截断采样，最终 RGB均值[0.4540,0.2283,-0.0886]，颜色和白底有改善，但图像仍模糊。
- 原CPU实验纯黑像素只占0.85%，没有精确重现附件的大面积纯黑。因此不能认定 x0 clipping 是该附件的唯一根因。
- 前16张训练图片 t999 噪声预测偏差（预测-真实）RGB=[-0.0514,-0.0005,+0.0636]，符合采样红偏高/蓝偏低方向。小样本诊断不代表泛化质量。
- 已用当前 torchvision 实测 autocontrast([-1,0,1])=[0,0.5,1]；训练增强范围存在独立问题。
- 参考：OpenAI improved-diffusion 官方 gaussian_diffusion.py 默认限制预测x0，再重算posterior mean；不要截断带噪x_t。

## 后续GPU复现
- 同输入CPU/GPU预测差异很小，最大约1.51e-5，不能将显卡兼容性警告当作已证实根因。
- GPU batch1 seed42生成偏亮图；同设置测试seed0..11发现黑/白漂移。选出最暗seed9：最终范围[-4.928,2.248]、63.83%通道值<-1、32.71%像素纯黑。已重现附件同类发黑现象，但未恢复附件原RNG状态。
- seed9同噪声x0截断对照改善背景、颜色和轮廓，仍模糊；不能宣称训练已充分。
- seed10有99.12%通道值>1，说明漂移也可向白色发展。
- 结果与候选修正写入diagnostics/README.md；原notebook、权重和原图保留，checkpoint SHA256校验一致。
