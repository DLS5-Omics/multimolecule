!!! warning "这是一个自动生成的翻译"

    本文内容为翻译版本，旨在为用户提供方便。
    我们已经尽力确保翻译的准确性。
    但请注意，翻译内容可能包含错误，仅供参考。
    请以英文[原文](license-faq.md)为准。

    为满足合规性与执法要求，翻译文档中的任何不准确或歧义之处均不具有约束力，也不具备法律效力。

# 许可协议常见问题解答

本许可协议常见问题解答（FAQ）阐明了由 DanLing 团队（也称为 DanLing）（“我们”、“我们”或“我们的”）提供的 MultiMolecule 项目（“MultiMolecule”）中数据、模型、代码、配置和文档的使用条款和条件。
本 FAQ 作为[GNU Affero 通用公共许可证（AGPL）](license.zh.md)（“许可证”）的附录，并以引用方式纳入其中。
本 FAQ 和许可协议共同构成您与我们之间关于您使用 MultiMolecule 的完整协议（“协议”）。
本 FAQ 中使用但未定义的大写术语具有 AGPL 中赋予它们的含义。

## 0. 要点摘要

本摘要重点介绍了我们许可的关键方面。有关更多详细信息，请参阅下面的相应部分并阅读[许可证](license.md)。

<div class="grid cards" markdown>

!!! question "在 MultiMolecule 中，什么构成了“源代码”？"

    MultiMolecule 中的源代码包括训练模型所需的所有材料，例如数据、代码、配置文件、文档和研究论文。模型权重被视为目标代码。

    [:octicons-arrow-right-24: 在 MultiMolecule 中，什么构成了“源代码”？](#1-multimolecule)

!!! question "我是否需要共享我微调的模型权重？"

    是的，任何源自 MultiMolecule 预训练模型的微调模型权重必须根据协议条款提供。

    [:octicons-arrow-right-24: 我是否需要共享我微调的模型权重？](#2)

!!! question "我需要致谢 MultiMolecule 吗？"

    是的，如果您在研究论文或其他项目中使用 MultiMolecule，则需要致谢 MultiMolecule。

    [:octicons-arrow-right-24: 我需要致谢 MultiMolecule 吗？](#3-multimolecule)

!!! question "我可以使用 MultiMolecule 发表研究论文吗？"

    视情况而定。您可以在某些开放获取和非盈利性场所发表研究论文。对于封闭获取或作者付费的场所，您必须获得单独的许可。

    [:octicons-arrow-right-24: 我可以使用 MultiMolecule 发表研究论文吗？](#4-multimolecule)

!!! question "在某些场所发表研究论文是否有任何限制？"

    是的，在某些场所发表研究论文有一些限制。

    [:octicons-arrow-right-24: 在某些场所发表研究论文的限制](#5)

!!! question "我可以将 MultiMolecule 用于商业目的吗？"

    是的，您可以根据协议条款将 MultiMolecule 用于商业目的。

    [:octicons-arrow-right-24: 我可以将 MultiMolecule 用于商业目的吗？](#6-multimolecule)

</div>

## 1. 在 MultiMolecule 中，什么构成了“源代码”？

在 MultiMolecule 的上下文中，“源代码”是指开发、训练和复制模型所需的所有材料。这包括但不限于：

- **数据**：用于训练和评估模型的数据集。
- **代码**：模型训练、评估和部署所需的所有脚本、程序和实用程序。
- **配置**：指示模型和训练过程行为的配置文件和设置。
- **文档**：便于理解和使用 MultiMolecule 的手册、指南和其他形式的文档。
- **研究论文**：描述与 MultiMolecule 相关的方法和发现的手稿和出版物。

相反，训练过程产生的 **模型权重** 被视为“目标代码”。

这种区分确保了训练和修改模型的所有必要组件保持可访问性，而训练后的模型权重被视为目标代码，这与我们致力于为社区提供最大自由度的承诺相一致。

## 2. 我是否需要共享我微调的模型？

是的，如果您使用了 MultiMolecule 模型中心的预训练模型并随后对其进行了微调，则必须根据协议条款开源您修改后的模型权重。

模型权重在 MultiMolecule 中被视为目标代码。因此，除了提供修改后的模型权重外，您还必须提供相应的源代码，如[许可证](license.md)第 1 节和本 FAQ 第 1 节中所定义和扩展的，以重现修改后的模型。这包括但不限于微调过程中使用的任何代码、数据、模型、配置文件和脚本，以及修改后的模型权重本身。

此外，如果您出于任何目的使用我们的相应源代码，您有义务根据协议条款提供所有相关材料，包括修改后的代码、派生数据集和微调模型。这种方法可确保所有增强功能、修改和相关资源对社区保持免费可用。

## 3. 我需要致谢 MultiMolecule 吗？

是的，您需要致谢 MultiMolecule，如果您在研究论文或其他项目中使用它。

!!! paper "研究论文"

    任何利用 MultiMolecule 资源的研究论文 _都应包含_ 对 MultiMolecule 项目的正式引用。具体的引用格式可能因出版场所的风格指南而异，但至少必须包含项目名称（“MultiMolecule”）和指向项目官方网站的链接（[https://multimolecule.danling.org](https://multimolecule.danling.org)）。如果由于场所限制无法进行正式引用，则 _应_ 在致谢部分提及 MultiMolecule 项目名称和 URL。

!!! code "项目"

    任何其他利用 MultiMolecule 组件的项目（包括软件、网络服务和其他衍生作品）_都应提供_ 对 MultiMolecule 项目的适当致谢。

    - **软件**：任何利用 MultiMolecule 组件的软件必须在其文档中包含 MultiMolecule 项目名称和 URL 的显著显示。此外，在每次启动生成的程序（或任何依赖于它的程序）时，_都应_ 向用户呈现 MultiMolecule 项目名称和 URL 的显著显示（例如，启动屏幕或横幅文本）。
    - **网络服务**：任何利用 MultiMolecule 组件的网络服务必须在服务的主页或其他易于访问的位置包含 MultiMolecule 项目名称和 URL 的显著显示。
    - **其他衍生作品**：任何其他衍生作品必须包含明确的致谢。

## 4. 我可以使用 MultiMolecule 发表研究论文吗？

任何使用 MultiMolecule 的出版物，无论出版场所如何，_必须包含正式引用_ MultiMolecule 项目。

!!! success "开放获取"

    您可以在钻石开放获取场所发表研究论文。

您可以在完全开放获取的期刊、会议或平台上发表研究论文，这些期刊、会议或平台不对作者或读者收取费用，前提是所有已发表的手稿均根据以下允许共享手稿的许可证之一提供：

- [GNU 自由文档许可证 (GFDL)](https://www.gnu.org/licenses/fdl.html)
- [知识共享许可证](https://creativecommons.org/licenses)
- [OSI 批准的许可证](https://opensource.org/licenses)

此许可是根据[许可证](license.md)第 7 节授予的额外许可。

!!! warning "非盈利"

    您可以在某些非盈利性场所发表研究论文。

您可以在某些非盈利性期刊、会议或平台上发表研究论文。具体来说，这包括：

- [美国科学促进会 (AAAS) 出版的所有期刊](https://www.aaas.org/journals)
- [eLife](https://elifesciences.org)

此许可是根据[许可证](license.md)第 7 节授予的额外许可。

!!! failure "封闭获取/作者付费"

    您必须获得单独的许可才能在封闭获取/作者付费场所发表。

我们不赞成在封闭获取或作者付费场所发表。在封闭获取或作者付费场所发表需要事先获得我们的单独书面许可协议。此类协议可能涉及共同作者身份或对项目的财务贡献等条件。

## 5. 在某些场所发表研究论文的限制

我们认为，免费和开放获取研究是机器学习社区的基石。受[《自然-机器智能》声明](https://openaccess.engineering.oregonstate.edu)以及持续的[零成本开放获取](https://diamasproject.eu)文化的启发，我们认为研究应该在没有作者或读者障碍的情况下普遍可访问。

以下出版场所采用封闭获取或作者付费模式，这与这些基本价值观相矛盾。我们认为这种做法是机器学习研究传播演变中的倒退，它破坏了社区促进开放协作和知识共享的努力：

- [《自然-机器智能》](https://www.nature.com/natmachintell)

我们强烈建议不要在上述场所发表使用 MultiMolecule 的作品，因为它们采用封闭获取或作者付费模式。

**尽管根据[许可证](license.md)第 7 节或通过与我们签订的任何单独书面许可协议授予了任何额外许可，但在上述场所发表 _未经授权_，除非该单独协议明确且具体地另有说明。** 此限制优先于此类额外许可或协议中的任何其他条款，并取代任何先前的弃权、豁免或许可，除非附加条款明确声明其优先于本节。

## 6. 我可以将 MultiMolecule 用于商业目的吗？

可以！您可以将 MultiMolecule 用于商业目的，前提是您遵守协议的所有条款。这包括根据协议条款提供 MultiMolecule 任何修改版本的相应源代码（如[许可证](license.md)第 1 节和本 FAQ 第 1 节中所定义）以及任何相关工件（包括您的训练数据和训练模型）的要求。

如果您希望将 MultiMolecule 用于商业目的而不根据[许可证](license.md)提供您的修改，则必须获得我们的单独书面许可协议。这可能涉及支持该项目的财务贡献。请通过 [multimolecule@zyc.ai](mailto:multimolecule@zyc.ai) 与我们联系以获取更多详细信息。

## 7. 隶属于某些组织的人员是否有特定的许可条款？

是的！如果您隶属于与我们签订了单独许可协议的组织，您可能需要遵守不同的许可条款。

对于隶属于以下组织的人员：

- [微软研究院](https://www.microsoft.com/en-us/research)
- [深势科技](https://dp.tech)
- [中关村人工智能研究院]

根据[许可证](license.md)第 7 节授予以下额外许可：

尽管有[许可证](license.md)第 13 节的规定，隶属于所列组织的人员无需向通过计算机网络远程与 MultiMolecule 修改版本交互的用户提供相应源代码，_前提是_ 此类使用仅限于其各自组织内的内部研究和开发目的。[许可证](license.md)的所有其他规定，包括但不限于如果您在组织外部发布 MultiMolecule 及其衍生作品则必须提供相应源代码的要求，仍然完全有效。此额外许可是不可转让且不可再许可的。

如果您 _不_ 隶属于所列组织之一，则这些额外许可 _不_ 适用于您，您必须遵守[许可证](license.md)的所有条款，包括第 13 节。您可能希望咨询您组织的法律部门，以确定是否存在单独的协议。

## 8. 如果我的组织禁止使用 AGPL 许可下的代码，我如何使用 MultiMolecule？

某些组织，例如 [Google](https://opensource.google/documentation/reference/using/agpl-policy)，禁止使用 AGPL 许可的代码。如果您隶属于一个不允许使用 AGPL 许可软件的组织，您必须获得我们的单独许可才能使用 MultiMolecule。

要请求单独的许可，请通过 [multimolecule@zyc.ai](mailto:multimolecule@zyc.ai) 与我们联系。

## 9. 如果我是美国政府的联邦雇员，我可以使用 MultiMolecule 吗？

不可以，美国政府的联邦雇员不能根据协议条款使用 MultiMolecule，因为美国联邦雇员以其官方身份编写的代码不受[美国法典第 17 卷第 105 条](https://www.law.cornell.edu/uscode/text/17/105)的版权保护。
因此，联邦雇员可能无法遵守协议条款。

## 10. 我们会更新此 FAQ 吗？

!!! tip "简而言之"

    是的，我们将根据需要更新此 FAQ，以保持符合相关法律。

我们可能会不时更新此许可 FAQ。
更新后的版本将在此许可 FAQ 底部的“上次修订时间”中注明。
如果我们进行任何重大更改，我们将通过在此页面上发布新的许可 FAQ 来通知您。
由于我们不收集您的任何联系信息，因此无法直接通知您。
我们鼓励您经常查看此许可 FAQ，以随时了解您如何使用我们的数据、模型、代码、配置和文档。
