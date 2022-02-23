import einops
import genops


def lbp_plus(
        unary_potentials,  # make yapf happy
        binary_potentials,
        binary_edges,
        ternary_potentials=None,
        ternary_edges=None,
        max_iter=10,
        damping=.5,
        tol=1e-5,
        track_best=False):
    genops.set_backend_as(unary_potentials)
    has_ternary = ternary_edges is not None
    if track_best:
        best_configuration = None
        best_energy = -10000
    # constants
    n_vertices, n_states = unary_potentials.shape
    n_bin_edges = binary_edges.shape[0]

    if has_ternary:
        n_ter_edges = ternary_edges.shape[0]

    directed_binaries = genops.stack([
        einops.rearrange(binary_potentials, "B E1 E2-> B E2 E1"),  # make yapf happy
        binary_potentials
    ])

    if has_ternary:
        directed_ternaries = genops.stack([
            einops.rearrange(ternary_potentials, "B E0 E1 E2-> B E1 E2 E0"),  # make yapf happy
            einops.rearrange(ternary_potentials, "B E0 E1 E2-> B E0 E2 E1"),
            ternary_potentials
        ])

    # variables to update
    all_incoming_msg = genops.zeros((n_vertices, n_states))
    last_bin_msg = genops.zeros((n_bin_edges, 2, n_states))
    if has_ternary:
        last_ter_msg = genops.zeros((n_ter_edges, 3, n_states))

    for _ in range(max_iter):
        diff = 0
        for ms, mt in ((0, 1), (1, 0)):
            src_incoming = all_incoming_msg[binary_edges[:, ms]]
            src_incoming_wo_factor = einops.rearrange(src_incoming - last_bin_msg[:, ms], "E S->E S 1")

            src_unary = einops.rearrange(unary_potentials[binary_edges[:, ms]], "E S->E S 1")

            s2t_potential = directed_binaries[mt]

            new_msg = src_incoming_wo_factor + src_unary + s2t_potential
            new_msg = einops.reduce(new_msg, "E S T-> E T", "max")
            new_msg -= einops.reduce(new_msg, "E T-> E 1", "max")
            new_msg = damping * last_bin_msg[:, mt] + (1 - damping) * new_msg
            delta_msg = new_msg - last_bin_msg[:, mt]
            last_bin_msg[:, mt] = new_msg

            delta_incoming = genops.sparse_coo_tensor(
                indices=genops.stack([binary_edges[:, mt], genops.arange(0, n_bin_edges)]),
                values=genops.ones([n_bin_edges]),
                shape=(n_vertices, n_bin_edges)
            ) @ delta_msg
            all_incoming_msg += delta_incoming

            diff += genops.abs(delta_msg).sum()

        if has_ternary:
            for ms1, ms2, mt in ((1, 2, 0), (0, 2, 1), (0, 1, 2)):
                s1_incoming = all_incoming_msg[ternary_edges[:, ms1]]
                s2_incoming = all_incoming_msg[ternary_edges[:, ms2]]
                s1_incoming_wo_factor = einops.rearrange(s1_incoming - last_ter_msg[:, ms1], "E S->E S 1 1")
                s2_incoming_wo_factor = einops.rearrange(s2_incoming - last_ter_msg[:, ms2], "E S->E 1 S 1")

                s1_unary = einops.rearrange(unary_potentials[ternary_edges[:, ms1]], "E S->E S 1 1")
                s2_unary = einops.rearrange(unary_potentials[ternary_edges[:, ms2]], "E S->E 1 S 1")

                to_t_potential = directed_ternaries[mt]

                new_msg = s1_incoming_wo_factor + s1_unary + s2_incoming_wo_factor + s2_unary + to_t_potential
                new_msg = einops.reduce(new_msg, "E S1 S2 T->E T", "max")
                new_msg = new_msg - einops.reduce(new_msg, "E T->E 1", "max")
                new_msg = damping * last_ter_msg[:, mt] + (1 - damping) * new_msg
                delta_msg = new_msg - last_ter_msg[:, mt]
                last_ter_msg[:, mt] = new_msg

                delta_incoming = genops.sparse_coo_tensor(
                    indices=genops.stack([ternary_edges[:, mt], genops.arange(0, n_ter_edges)]),
                    values=genops.ones([n_ter_edges]),
                    shape=(n_vertices, n_ter_edges)
                ) @ delta_msg
                all_incoming_msg += delta_incoming
                diff += genops.abs(delta_msg).sum()
        if track_best:
            configuration = genops.argmax(all_incoming_msg + unary_potentials, axis=1)
            energy = compute_energy_plus(
                unary_potentials=unary_potentials,  # yapf
                binary_potentials=binary_potentials,
                binary_edges=binary_edges,
                ternary_potentials=ternary_potentials,
                ternary_edges=ternary_edges,
                labels=configuration)
            if energy > best_energy:
                best_energy = energy
                best_configuration = configuration

        if diff < tol:
            break
    if track_best:
        return best_configuration
    else:
        return genops.argmax(all_incoming_msg + unary_potentials, axis=1)


def compute_energy_plus(
        unary_potentials,  #
        binary_potentials,
        binary_edges,
        ternary_potentials=None,
        ternary_edges=None,
        labels=None):
    energy = genops.sum(unary_potentials[genops.arange(len(labels)), labels])

    # for edge, pw in zip(binary_edges, binary_potentials):
    #     energy += pw[labels[edge[0]], labels[edge[1]]]
    n_state = binary_potentials.shape[-1]
    score = einops.rearrange(binary_potentials, "B S1 S2->B (S1 S2)")
    score_index = labels[binary_edges]
    score_index = score_index[:, 0] * n_state + score_index[:, 1]
    score_index = einops.rearrange(score_index, "B->B ()")
    bin_score = genops.gather(score, axis=1, index=score_index)
    energy += genops.sum(bin_score)

    if ternary_potentials is not None:
        # for edge, tw in zip(ternary_edges, ternary_potentials):
        #     energy += tw[labels[edge[0]], labels[edge[1]], labels[edge[2]]]
        score = einops.rearrange(ternary_potentials, "B S1 S2 S3->B (S1 S2 S3)")
        score_index = labels[ternary_edges]
        score_index = score_index[:, 0] * (n_state**2) + score_index[:, 1] * n_state + score_index[:, 2]
        score_index = einops.rearrange(score_index, "B->B ()")
        ter_score = genops.gather(score, axis=1, index=score_index)
        energy += genops.sum(ter_score)

    return energy
