!!! warning "这是一个自动生成的翻译"

    本文内容为翻译版本，旨在为用户提供方便。
    我们已经尽力确保翻译的准确性。
    但请注意，翻译内容可能包含错误，仅供参考。
    请以英文[原文](license-faq.md)为准。

    为满足合规性与执法要求，翻译文档中的任何不准确或歧义之处均不具有约束力，也不具备法律效力。

# 许可协议常见问题解答

本许可协议常见问题解答（Frequently Asked Questions，FAQ）用于澄清 DanLing 团队（亦称 DanLing）（“我们”、“我方”或“我们的”）在 MultiMolecule 项目（“MultiMolecule”）中提供之材料的使用条款与条件。
本 FAQ 作为 [GNU Affero 通用公共许可证（AGPL）](license.md)（“许可证（License）”）的补充文件，并以引用方式并入该许可证。
本 FAQ 与许可证共同构成你与我们就使用 MultiMolecule 所达成的完整协议（“协议（Agreement）”）。
本 FAQ 中使用但未定义之首字母大写术语，具有 AGPL 中赋予之含义。

## 0. 要点摘要 {#0-summary-of-key-points}

本节概述本许可证的关键要点。
如需更详细信息，请参阅下文对应章节并阅读[许可证](license.md)。

<div class="grid cards" markdown>

!!! question "MultiMolecule 中的源代码与目标代码是什么？"

    源代码（Source Code）包括开发、训练、评估与运行模型所需的全部材料，包括数据、代码、配置与文档。
    目标代码（Object Code）包括模型权重文件与编译后的代码。

    [:octicons-arrow-right-24: MultiMolecule 中的源代码与目标代码是什么？](#1-what-are-source-code-and-object-code-in-multimolecule)

!!! question "我是否必须共享我训练得到的模型？"

    若你传播（Convey）由 MultiMolecule 托管并分发的模型权重文件，则你必须在协议条款下传播这些权重文件，并同时提供相应源代码（Corresponding Source）。
    若你传播上述权重文件的修改版本（例如微调后的权重），则你必须在协议条款下传播这些修改后的权重文件，并同时提供相应源代码（Corresponding Source）。
    无论你是使用 MultiMolecule、第三方库，或自定义训练流水线来生成被传播的权重，上述义务均适用。

    [:octicons-arrow-right-24: 我是否必须共享我训练得到的模型？](#2-am-i-required-to-share-my-trained-model)

!!! question "我是否必须共享用于训练的数据？"

    若你传播第 2 节所涵盖的任何模型权重，则你还必须在协议条款下向接收者提供用于训练、更新或修改该等权重的任何训练数据集，并排除仅用于评估的数据（详见第 3 节的澄清）。

    [:octicons-arrow-right-24: 我是否必须共享用于训练的数据？](#3-am-i-required-to-share-the-data-used-for-training)

!!! question "我是否需要致谢 MultiMolecule？"

    当 MultiMolecule 对你的工作有贡献时，我们强烈鼓励你进行致谢。
    我们强烈建议所有研究论文进行引用，但仅在你依赖第 5 节或第 8 节所授予的额外许可时，引用才构成该额外许可的强制条件。
    在某些已分发的交互式界面中，适当的法律声明（Appropriate Legal Notices）中需要保留合理的作者署名信息（author attribution）。

    [:octicons-arrow-right-24: 我是否需要致谢 MultiMolecule？](#4-do-i-need-to-acknowledge-multimolecule)

!!! question "我可以使用 MultiMolecule 发表研究论文吗？"

    若你的手稿或补充材料包含 MultiMolecule 材料（如第 5 节所述），则除非适用额外许可，否则手稿与补充材料必须在许可证条款下分发。
    第 5 节与第 8 节在第 6 节约束下授予额外许可，以便在特定场所依据“手稿共享许可”发表。
    若你无法遵守许可证且不满足额外许可条件，则你不得在包含 MultiMolecule 材料的情况下分发该手稿或补充材料。

    [:octicons-arrow-right-24: 我可以使用 MultiMolecule 发表研究论文吗？](#5-can-i-publish-research-papers-using-multimolecule)

!!! question "在某些出版场所发表研究论文是否存在限制？"

    是的，本 FAQ 授予的额外许可在某些出版场所有相应限制。
    第 6 节仅限制本 FAQ 授予的额外许可。
    第 6 节不限制你在[许可证](license.md)本身框架下的发表。

    [:octicons-arrow-right-24: 在某些出版场所发表研究论文的限制](#6-restrictions-on-publishing-research-papers-in-certain-venues)

!!! question "我可以将 MultiMolecule 用于商业目的吗？"

    可以，你可以在协议条款下将 MultiMolecule 用于商业目的。
    若你希望在不承担传播受覆盖材料时所适用义务的情况下进行商业使用，你必须获得我们单独授予的许可。

    [:octicons-arrow-right-24: 我可以将 MultiMolecule 用于商业目的吗？](#7-can-i-use-multimolecule-for-commercial-purposes)

!!! question "MultiMolecule 协作者是否有特殊许可？"

    是的，经认可的协作者将依据许可证第 7 节获得特定额外许可。
    这些许可受所述条件与第 6 节约束。

    [:octicons-arrow-right-24: MultiMolecule 协作者是否有特殊许可？](#8-are-there-special-permissions-for-multimolecule-collaborators)

</div>

## 1. 在 MultiMolecule 中，什么是源代码和目标代码？ {#1-what-are-source-code-and-object-code-in-multimolecule}

对于 MultiMolecule 项目中的所有材料，下列定义用于澄清并补充[许可证](license.md)中的定义。

> [!TIP] MultiMolecule 托管材料的范围
> 除非在相关模型卡、数据集卡、文件头、目录说明或随附的 LICENSE/NOTICE 中另有明确声明，否则所有由 MultiMolecule 托管并分发的模型权重、数据集、代码、配置与文档均在本协议下提供。
> 若我们未来以不同条款托管任何特定项目，我们将对该项目作出明确标注，并以标注条款为准。

> [!IMPORTANT] 源代码（Source Code）
> **源代码（Source Code）** 是指受许可材料用于进行修改的首选形式，且与许可证第 1 条保持一致。
> 源代码（Source Code）涵盖开发、训练、评估与运行模型所需的全部材料。

源代码（Source Code）包括但不限于：

- **数据（Data）**：以处理所需的形式提供的、训练、评估或运行受许可材料所提供或生成模型所需的数据集。
- **代码（Code）**：处理数据、训练模型、执行评估、部署模型或以其他方式运行与修改受许可材料所需的全部源代码脚本、程序、库（包括模型结构与流水线定义）及工具。
- **配置（Configuration）**：与受许可材料相关的安装、编译、训练、评估、运行或执行过程所需的配置文件、参数设置、环境规范及控制脚本。
- **文档（Documentation）**：接口定义文件、构建说明、手册、指南、作为受许可材料一部分由 MultiMolecule 分发的研究论文与技术报告（用于描述具体方法、结构与参数），以及理解、安装、运行与修改受许可材料所需的其他技术文档。

*按上述定义提供源代码（Source Code），是在传播目标代码（Object Code）时满足协议项下提供“相应源代码（Corresponding Source）”要求所必需的（在适用情况下亦用于满足许可证项下之对应要求）。*

> [!IMPORTANT] 目标代码（Object Code）
> **目标代码（Object Code）** 是指不属于源代码（Source Code）的任何形式之受许可材料。

目标代码（Object Code）主要包括但不限于：

- **模型权重（Model Weights）**：训练后表征模型学习状态的数值参数（例如 SafeTensors、HDF5 或类似格式文件）。
  这包括除非另有声明之外所有由 MultiMolecule 提供或托管的模型权重。
  这也包括从由 MultiMolecule 提供或托管的模型权重派生的任何微调模型权重。
- **编译代码（Compiled Code）**：非人类可读的可执行软件代码，例如 Python 包中可能包含的已编译 C++ 扩展。

*对于在 MultiMolecule 中被视为目标代码（Object Code）的模型权重，相应源代码（Corresponding Source）至少包括复现被传播权重所需的训练数据与脚本。*

理解上述区分有助于澄清你在本协议项下的义务。
例如，若你传播目标代码（Object Code）（如模型权重），则你还必须确保相应的源代码（Source Code）（包括必要的数据、代码、配置与文档）亦在协议条款下可获得。

## 2. 我是否必须共享我训练得到的模型？ {#2-am-i-required-to-share-my-trained-model}

若你传播本协议所涵盖的模型权重，则你必须在本协议条款下传播这些权重，并同时提供相应源代码（Corresponding Source）。

如本 FAQ 第 1 节所述，模型权重在 MultiMolecule 中被视为目标代码（Object Code）。
每当你依据本协议传播目标代码（Object Code）时，你亦必须提供相应源代码（Corresponding Source）。
对于受本协议覆盖的模型权重，相应源代码（Corresponding Source）至少包括复现、安装、运行与修改被传播权重所需的代码、训练数据、配置与脚本，并如本 FAQ 第 1 节所澄清。

若你修改 MultiMolecule 且你通过计算机网络向用户提供对你修改版本的远程交互，则你必须遵守许可证第 13 节。
你必须向与该修改版本远程交互的所有用户提供获得你该修改版本相应源代码（Corresponding Source）的机会。
本第 13 节义务所指向的是对“修改后的 MultiMolecule 程序（Program）”本身的远程交互。

第 3 节进一步规定训练数据要求。

## 3. 我是否必须共享用于训练的数据？ {#3-am-i-required-to-share-the-data-used-for-training}

若你传播本协议所涵盖的模型权重，则你还必须在本协议条款下向接收者提供用于训练、更新或修改该等权重的任何训练数据集，并排除仅用于评估的数据（详见下文澄清）。

如本 FAQ 第 1 节所述，训练数据集在 MultiMolecule 中被视为源代码（Source Code）。
因此，当你依据本协议需要提供相应源代码（Corresponding Source）时，所需相应源代码（Corresponding Source）应包含复现被传播权重所需的训练数据，并如本 FAQ 第 1 节所澄清。

该要求仅适用于用于训练、更新或修改被传播模型权重的数据。
仅用于评估的数据不在本条要求之列，前提是该数据未同时用于训练。

若所需训练数据无法在本协议项下向接收者提供，则你不得在本协议项下传播相应权重。

## 4. 我是否需要致谢 MultiMolecule？ {#4-do-i-need-to-acknowledge-multimolecule}

当 MultiMolecule 对你的工作有贡献时，我们强烈鼓励你进行致谢。
本节区分以下两类内容：

- (a) 作为社区规范我们强烈请求的内容，以及
- (b) 仅在你依赖本 FAQ 授予的额外许可时才成为强制条件的内容。

> [!NOTE]
> 我们强烈建议任何使用 MultiMolecule 的研究论文对 MultiMolecule 进行正式引用。
> 若你仅依据[许可证](license.md)发表论文，则引用不是许可证合规条件，但我们强烈请求你进行引用。
> 若你依赖第 5 节或第 8 节的额外许可，则正式引用是该额外许可的条件。

当本 FAQ 要求引用时（例如作为第 5 节或第 8 节额外许可的条件），引用至少必须包含项目名称（“MultiMolecule”）与 DOI（[10.5281/zenodo.12638419](https://doi.org/10.5281/zenodo.12638419)）。
若出版场所不支持正式引用，则项目名称与 DOI 必须改为出现在致谢部分。

> [!IMPORTANT]
> 若你传播包含 MultiMolecule 的程序，你必须在适当的法律声明（Appropriate Legal Notices）中保留对 MultiMolecule 的合理作者署名信息。

若程序具有交互式用户界面，则其必须展示适当的法律声明（Appropriate Legal Notices）。
该等声明必须包含对 MultiMolecule 的合理作者署名信息，包括项目名称与官方仓库或网站链接。

对于命令行程序，该署名应在启动时显著展示，并在适用情况下可通过 `--help`、`--version` 或 `--about` 获取。
对于网络服务，该署名应在主页或其他易于访问的位置显著展示。

对于库或非交互组件，该署名应在文档中显著展示，并在组件提供任何交互界面时在该界面中显著展示。

## 5. 我可以使用 MultiMolecule 发表研究论文吗？ {#5-can-i-publish-research-papers-using-multimolecule}

> [!IMPORTANT]
> 如第 1 节所述，文档（Documentation）属于 MultiMolecule 源代码（Source Code）的一部分。
> 因此，若你的手稿或补充材料包含 MultiMolecule 材料（即包含或复制 MultiMolecule 的代码、权重、数据集、文档文本、图表或其他 MultiMolecule 材料），则该手稿与补充材料将被视为与这些 MultiMolecule 材料一并分发的文档（Documentation）。
> 该要求仅适用于“包含 MultiMolecule 材料”的手稿与补充材料，而不适用于仅与 MultiMolecule 一同分发但彼此独立的作品。
> 因此，在无额外许可的情况下，在手稿或补充材料包含或复制 MultiMolecule 材料的范围内，手稿与补充材料必须作为同一份 MultiMolecule 材料分发的一部分，依据许可证条款进行分发。
> 若你无法遵守许可证，则你不得在本协议项下于包含 MultiMolecule 材料的情况下分发该手稿或补充材料。

本节依据许可证第 7 节授予额外许可，用于作者希望在“手稿共享许可”下发布手稿的特定出版场景。
为避免疑义，本第 5 节中的额外许可仅适用于手稿文本与其他文档类材料。
该等额外许可不改变你所传播的任何 MultiMolecule 代码、模型权重或数据集所适用的许可证。
除非另有明确声明，否则你所传播的上述 MultiMolecule 材料仍受本协议约束。

> [!IMPORTANT]
> 若你依赖本第 5 节授予的任何额外许可，则第 4 节所述正式引用要求是该额外许可的条件。
> 本第 5 节授予的任何额外许可仍受第 6 节约束。

> [!TIP] 钻石开放获取（Diamond Open Access）
> 下述额外许可允许在钻石开放获取场所发表。

你可以在完全开放获取的期刊、会议或平台发表传播 MultiMolecule 材料的手稿。
该等场所不得向作者或读者收取费用。

手稿的公开版本必须以允许共享手稿的许可证发布。
你可以使用下列许可证之一。

- GNU 自由文档许可证（GFDL）
- 知识共享许可证（Creative Commons）
- OSI 认可许可证

该许可作为许可证第 7 节项下的额外许可授予。

> [!WARNING] 非营利（Non-Profit）
> 下述额外许可允许在特定非营利场所发表。

你可以在某些非营利期刊、会议或平台发表传播 MultiMolecule 材料的手稿。
该范围包括如下场所。

- eLife

该许可作为许可证第 7 节项下的额外许可授予。

> [!CAUTION] 封闭获取 / 作者付费（Closed-Access / Author-Fee）
> 封闭获取或作者付费场所往往使合规变得不可能。

我们不赞成在封闭获取或作者付费场所发表 MultiMolecule 材料。

若某出版场所的条款将阻止你就与发表相关而传播的任何 MultiMolecule 材料遵守许可证，则你必须在投稿或发表前与我们达成单独的书面许可协议。
该等协议可能包含共同作者身份或对项目的资金支持等条件。

## 6. 在某些出版场所发表研究论文的限制 {#6-restrictions-on-publishing-research-papers-in-certain-venues}

> [!IMPORTANT]
> 本节仅限制本 FAQ 授予的额外许可。
> 本节不限制你在[许可证](license.md)本身框架下的发表。
> 若你在发表中传播 MultiMolecule 材料并就该等材料遵守许可证，则第 6 节不适用。

因此，第 6 节仅约束第 5 节与第 8 节中的额外许可。
第 6 节亦约束任何引用或并入本 FAQ 的单独书面许可协议，除非该单独协议以书面形式明确声明不适用本节。

我们认为，免费和开放获取研究是机器学习社区的基石。
受《自然·机器智能》声明与持续的零成本开放获取文化启发，我们认为研究应该在没有作者或读者障碍的情况下普遍可访问。

以下出版场所采用封闭获取或作者付费模式，这与这些基本价值观相矛盾。
我们认为这种做法是机器学习研究传播演变中的倒退，会削弱社区推动开放协作与知识共享的努力。

- Nature Machine Intelligence

尽管有第 5 节与第 8 节的规定，本 FAQ 授予的任何额外许可均不授权你在上述场所投稿或发表传播 MultiMolecule 材料的手稿。
只有在你与我们另行签署单独书面许可协议，且该协议以明确条款允许你在不受本第 6 节限制的情况下发表时，你才可以在上述场所投稿或发表该等手稿。

我们强烈建议不要在上述场所发表传播 MultiMolecule 材料的作品。

## 7. 我可以将 MultiMolecule 用于商业目的吗？ {#7-can-i-use-multimolecule-for-commercial-purposes}

可以。
你可以在本协议条款下将 MultiMolecule 用于商业目的，前提是你遵守本协议。

若你传播经修改的 MultiMolecule 材料，则你必须提供许可证与本 FAQ 所要求的相应源代码（Corresponding Source）及相关工件。

在适用情况下，这包括第 3 节所澄清的训练数据与第 2 节所澄清的模型权重。

若你希望在商业使用中不按许可证公开上述材料，则你必须与我们达成单独的书面许可协议。
请通过 [license@danling.org](mailto:license@danling.org) 联系我们以获取更多信息。

## 8. MultiMolecule 协作者是否有特殊许可？ {#8-are-there-special-permissions-for-multimolecule-collaborators}

是的。
若你被 DanLing 团队认可为协作者，则你有权依据[许可证](license.md)第 7 节获得以下额外许可。

> [!TIP] 内部网络使用豁免
> 尽管有许可证第 13 节的规定，协作者在其团队内部研发场景中使用经修改的 MultiMolecule 版本并通过计算机网络供团队成员远程交互时，可豁免向该等用户提供相应源代码（Corresponding Source）的义务。
> 该豁免不适用于外部用户、公开部署或传播（Conveyance）。

> [!TIP] 论文发表的扩展许可
> 协作者可以在任何同行评审的科学出版场所发表传播 MultiMolecule 材料的手稿，包括期刊与会议论文集，而不受访问模式或作者费用影响。
> 该扩展许可作为许可证第 7 节项下的额外许可授予。
> 该扩展许可仍受本 FAQ 第 6 节约束。
> 该扩展许可仅影响手稿与补充文档的许可方式，而不改变你所传播的任何 MultiMolecule 材料所适用的许可证。
> 作为该扩展许可的条件，你必须遵守第 4 节的致谢与引用要求。

> [!IMPORTANT] 与出版相关的源代码发布时间
> 该额外许可仅涉及与出版相关修改的公开发布时间点。
> 该额外许可并不延后你在传播时向接收者提供相应源代码的义务，也不延后你在第 13 节项下对网络远程交互用户提供相应源代码的义务。
> 若你的修改被用于手稿所描述的研究，则你必须在以下事件中最先发生者发生时，公开发布与该出版相关修改的相应源代码（Corresponding Source）。
>
> - 手稿在同行评审场所被正式接收发表。
> - 自手稿首次发布于公共预印本服务器起已过去 366 天。
>
> 你必须在首个适用触发事件发生时立即进行公开发布。
> 若修改的传播或网络远程交互与出版或预印本无关，则适用许可证的标准时间规则。

> [!NOTE] 协作者额外许可的一般条件
> 上述额外许可仅授予由 DanLing 团队认可的活跃、受邀协作者。
> 上述额外许可不可转让且不可再许可。
> 除非上文明确修改，否则许可证与本 FAQ 的其他条款仍完全有效。
> DanLing 团队可通过书面沟通按个案授予额外许可。

## 9. 如果我的组织禁止使用 AGPL 许可下的代码，我如何使用 MultiMolecule？ {#9-how-can-i-use-multimolecule-if-my-organization-forbids-the-use-of-code-under-the-agpl-license}

某些组织（例如 [Google](https://opensource.google/documentation/reference/using/agpl-policy)）禁止使用 AGPL 许可的代码。
若你所属组织不允许使用 AGPL 许可的软件，则你必须获得我们的单独许可方可使用 MultiMolecule。
如需申请单独许可，请通过 [license@danling.org](mailto:license@danling.org) 联系我们。

## 10. 我们会更新此 FAQ 吗？ {#10-do-we-make-updates-to-this-faq}

> [!TIP] 简而言之
> 是的，我们将根据需要更新本 FAQ，以保持对相关法律的合规。

我们可能不时更新本许可 FAQ。
更新版本将通过本页底部“最后修订时间（Last Revised Time）”的更新予以标示。
若我们作出任何重大变更，我们将通过在本页面发布新的许可 FAQ 的方式进行通知。
由于我们不会收集你的联系信息，我们无法向你进行直接通知。
我们建议你定期查阅本许可 FAQ，以了解你如何使用我们的数据、模型、代码、配置与文档。
