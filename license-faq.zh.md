!!! warning "翻译"

    本文内容为翻译版本，旨在为用户提供方便。
    我们已经尽力确保翻译的准确性。
    但请注意，翻译内容可能包含错误，仅供参考。
    请以英文[原文](license-faq.md)为准。

    为满足合规性与执法要求，翻译文档中的任何不准确或歧义之处均不具有约束力，也不具备法律效力。

# 许可协议常见问题解答

本许可常见问题解释了 DanLing 团队（亦称为DanLing）("我们"、"我们"或"我们的")提供的数据、模型、代码、配置和文档的使用条款和条件。它作为我们的 _[许可协议](license.zh.md)_ 的补充。本 许可协议常见问题解答 和 _[许可协议](license.zh.md)_ 作为一个完整 **协议** 向想要获得授权的人授权。

## 0. 关键要点摘要

本摘要突出显示了我们许可协议的关键方面。有关更详细的信息，请参阅下面的相应章节和阅读 _[许可协议](license.zh.md)_ 。

<div class="grid cards" markdown>

!!! question " MultiMolecule 中的“源代码”包括哪些内容？"

     MultiMolecule 中的源代码包括训练模型所需的所有材料，如数据、代码、配置文件、文档和研究论文。模型权重被视为目标代码。

    [:octicons-arrow-right-24:  MultiMolecule 中的“源代码”包括哪些内容？](#1-multimolecule)

!!! question "我需要开源我微调后的模型权重吗？"

    是的，任何使用 MultiMolecule 的预训练模型并随后进行微调的模型权重必须在 **协议** 下开源。

    [:octicons-arrow-right-24: 我需要开源我微调后的模型权重吗？](#2)

!!! question "我可以使用 MultiMolecule 发表研究论文吗？"

    视情况而定。你可以在某些开放获取和非盈利场所发表研究论文。对于封闭获取或作者费用场所，你必须获得单独的许可。

    [:octicons-arrow-right-24: 我可以使用 MultiMolecule 发表研究论文吗？](#3-multimolecule)

!!! question "我可以在 Nature Machine Intelligence 上 发表 MultiMolecule 的衍生工作吗？"

    不能。MultiMolecule 不得用于提交至或发表在 Nature Machine Intelligence 的任何作品中。

    [:octicons-arrow-right-24: 在 Nature Machine Intelligence 发表](#4-nature-machine-intelligence)

!!! question "我可以将 MultiMolecule 用于商业目的吗？"

    是的，你可以根据 **协议** 的条款将 MultiMolecule 用于商业目的。

    [:octicons-arrow-right-24: 我可以将 MultiMolecule 用于商业目的吗？](#5-multimolecule)

!!! question "与某些组织有关系的人是否有特定的许可条款？"

    是的，与某些组织有关系的人有特定的许可条款。

    [:octicons-arrow-right-24: 与某些组织有关系的人是否有特定的许可条款？](#6)

</div>

## 1.  MultiMolecule 中的“源代码”包括哪些内容？

在 MultiMolecule 的上下文中，“源代码”指的是训练和开发模型所需的所有材料。这包括：

- **数据**：用于训练和评估模型的数据集。
- **代码**：所有用于模型训练、评估和部署的脚本、程序和工具。
- **配置**：配置文件和设置，这些文件和设置决定了模型和训练过程的行为。
- **文档**：手册、指南以及其他有助于理解和使用软件的文档形式。
- **研究论文**：描述与 MultiMolecule 相关的方法和发现的手稿和出版物。

相反，训练过程中生成的**模型权重**被视为“目标代码”。

这种区分确保了训练和修改模型所需的所有必要组件都是可访问的，而训练后的模型权重则被视为目标代码，这与我们为社区提供最大自由度的承诺一致。

## 2. 我是否需要开源我微调后的模型？

是的，如果您使用了来自 MultiMolecule 模型中心的预训练模型并对其进行了微调，您必须根据 **协议** 开源您微调后的模型权重。

在 MultiMolecule 中，模型权重被视为目标代码（object code）。因此，除了分享微调后的模型权重，您还必须开源所有用于再现该模型的必要材料，包括训练数据、配置文件以及任何用于微调的脚本。

进一步的，如果你使用了我们的代码、数据或者模型进行任何目的，您有义务根据 **协议** 开源所有相关的材料，包括微调后的模型、修改后的代码或派生数据集。这种做法确保了所有增强、修改和相关资源对社区保持自由可用。

## 3. 我可以使用 MultiMolecule 发表研究论文吗？

!!! success "开放获取"

    你可以在钻石开放获取场所发表研究论文。

你被允许在不向作者或读者收取费用的完全开放获取的期刊、会议或平台上发表研究论文，前提是所有发表的手稿均以以下许可协议之一下提供以允许共享手稿：

- [GNU 自由文档许可证 (GFDL)](https://www.gnu.org/licenses/fdl.html)
- [Creative Commons 许可证](https://creativecommons.org/licenses)
- [OSI 认证许可证](https://opensource.org/licenses)

!!! warning "非盈利"

    你可以在某些非营利场所上发表研究论文。

你被允许在某些非盈利期刊、会议或平台上发表研究论文。具体的，这包括：

- [美国科学促进会 (AAAS) 主办的所有期刊](https://www.aaas.org/journals)
- [eLife](https://elifesciences.org)

此许可作为 _[许可协议](license.zh.md)_ 第7条的特别豁免被授予。

!!! failure "封闭获取/作者费用"

    你必须获得单独的许可才能在封闭获取或作者费用场所上发表论文。

我们不支持在封闭获取或收取作者费用的场所发表。要在这些场所发表，你必须从我们这里获得单独的许可。这通常涉及以下一项或多项：

- 与 DanLing 团队共同署名。
- 支持项目的费用。

虽然不是强制性的，我们建议你在研究论文中引用 MultiMolecule 项目。

## 4. 在 Nature Machine Intelligence 发表

我们相信，自由与开放获取研究成果是机器学习社区的基石。受到[有关 Nature Machine Intelligence 的声明](https://openaccess.engineering.oregonstate.edu)的启发，以及在推进的[零成本开放获取](https://diamasproject.eu)文化的鼓舞，我们坚信研究应当在不对作者或读者设置门槛的情况下普及与传播。

[《Nature Machine Intelligence》](https://www.nature.com/natmachintell)采用闭源访问或作者付费模式，这违背了这些基本价值观。我们认为，这样的做法是对机器学习研究成果传播方式的倒退，破坏了社区一直以来为促进开放合作与知识共享所付出的努力。

MultiMolecule 严格禁止在提交给《Nature Machine Intelligence》或在其上发表的作品中使用我们的代码、数据、模型或相关作品。此禁令的效力高于任何先前授予的豁免、特许或许可。除非有额外条款明确声明可取代本条规定，否则在《Nature Machine Intelligence》上使用 MultiMolecule 的任何行为均不被允许。

## 5. 我可以将 MultiMolecule 用于商业目的吗？

是的！你可以根据 **协议** 将 MultiMolecule 用于商业用途。注意，你必须开源对源代码的任何修改，并使其在 **协议** 下可用。

如果你希望在不开源修改内容的情况下将 MultiMolecule 用于商业用途，则必须从我们这里获得单独的许可。这通常涉及支持项目的费用。请通过 [multimolecule@zyc.ai](mailto:multimolecule@zyc.ai) 与我们联系以获取更多详细信息。

## 6. 与特定组织相关联的人是否有特定的许可条款？

是的！如果你与一个与我们有单独许可协议的组织相关联，你可能会受到不同的许可条款的约束。

以下组织的成员自动获得一个不可再分发的 [MIT 许可协议](https://mit-license.org) 以使用 MultiMolecule：

- [微软科学智能研究院](https://www.microsoft.com/en-us/research/lab/microsoft-research-ai-for-science)
- [深势科技](https://dp.tech)

此许可作为 _[许可协议](license.zh.md)_ 第7条的特别豁免被授予。主要要点包括：

- **不可再分发**：授予这些组织成员的MIT许可是不可转让和不可再许可的。
- **禁止独立派生作品**：你被禁止基于此许可创建独立的派生作品。
- **遵守AGPL许可**：基于此许可的任何修改或派生作品自动被视为 MultiMolecule 的派生作品，必须遵守 **协议** 的所有条款。这确保了第三方无法绕过许可条款或通过派生作品建立单独的许可。

如果你未与上述列出的组织相关联，请咨询你组织的法律部门，以确定你是否需遵守与我们签订的单独许可协议。

## 7. 如果我的组织禁止使用AGPL许可的代码，我该如何使用 MultiMolecule ？

某些组织，如[谷歌](https://opensource.google/documentation/reference/using/agpl-policy)，禁止使用AGPL许可的代码。如果你隶属于禁止使用AGPL许可软件的组织，你必须从我们这里获得单独的许可才能使用 MultiMolecule 。

要申请单独的许可，请通过[电子邮件](mailto:multimolecule@zyc.ai)与我们联系。

## 8. 我是美国联邦政府的联邦雇员，我可以使用 MultiMolecule 吗？

不可以，美国联邦政府的联邦雇员无法根据 **协议** 使用 MultiMolecule ，因为根据[17 U.S. Code § 105](https://www.law.cornell.edu/uscode/text/17/105)规定，由美国联邦雇员撰写的代码不受版权保护。

因此，联邦雇员无法遵守 **协议** 的条款，因此无法使用 MultiMolecule 。

## 9. 我们会更新此常见问题吗？

!!! tip "简而言之"

    是的，我们将根据需要更新此常见问题解答以保持与相关法律的一致。

我们可能会不时更新此许可协议常见问题解答。
更新后的版本将通过更新本页面底部的“最后修订时间”来表示。
如果我们进行任何重大更改，我们将通过在本页发布新的许可协议常见问题解答来通知你。
由于我们不收集你的任何联系信息，我们无法直接通知你。
我们鼓励你经常查看本许可协议常见问题解答，以了解你可以如何使用我们的数据、模型、代码、配置和文档。
