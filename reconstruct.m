function [x, info] = reconstruct(prob)
	t_st  = tic;
	t_lim = 35.0;

	b   = prob.b(:);
	m   = length(b);

	N   = prob.N;
	ang = prob.angles;
	ndt = prob.n_det;

	xm  = 0;
	xM  = 1;
	sig = 0;

	if isfield(prob, 'x_min'), xm  = prob.x_min;  end
	if isfield(prob, 'x_max'), xM  = prob.x_max;  end
	if isfield(prob, 'sigma'), sig = prob.sigma;  end

	delta = 0.05 * (xM - xm);
	d_tgt = sqrt(m) * sig;
	ratio = m / (N * N);

	if N >= 256
		levels	= [N/4, N/2, N];
		iters_per = [6,   9,  40];
	elseif N >= 128 && ratio < 0.4
		levels	= [N/2, N];
		iters_per = [6,  37];
	else
		levels	= [N];
		iters_per = [24];
	end

	x		 = [];
	prev_N	= 0;
	tot_iters = 0;
	is_smooth = false;

	for L = 1:length(levels)
		N_cur = levels(L);

		if L == 1
			x = fbp_init(b, ang, N_cur, ndt);

			if ratio >= 0.4
				[gx0, gy0] = grad2D(reshape(x, N_cur, N_cur));
				gm0   = sqrt(gx0.^2 + gy0.^2);
				K	 = numel(gm0);
				L1g   = sum(gm0(:));
				L2g   = sqrt(sum(gm0(:).^2) + eps);
				hoyer = (sqrt(K) - L1g / L2g) / (sqrt(K) - 1);
				is_smooth = (hoyer < 0.27);
			end
		else
			x = upsample_image(x, prev_N, N_cur);
		end
		x = max(xm, min(xM, x));

		is_last = (L == length(levels));
		[x, k]  = solve_level(x, b, ang, N_cur, ndt, sig, ...
							  xm, xM, delta, d_tgt, ...
							  iters_per(L), t_st, t_lim, is_last, is_smooth);

		tot_iters = tot_iters + k;
		prev_N	= N_cur;

		if toc(t_st) > t_lim - 0.5
			break;
		end
	end

	info.iters = tot_iters;
	info.time  = toc(t_st);
end

function [x, out_it] = solve_level(x, b, ang, N, ndt, sig, ...
								   xm, xM, delta, d_tgt, ...
								   max_iters, t_st, t_lim, is_last, is_smooth)

	ATb = adjoint_op(b, ang, N, ndt);
	scA = max(1, max(abs(ATb)));

	m	 = length(b);
	ratio = m / (N * N);

	if is_smooth
		C_lam	  = 500;
		gamma_coef = 0;
	elseif N < 128
		C_lam	  = 1000;
		gamma_coef = 100;
	elseif N < 256
		if ratio < 0.4
			C_lam	  = 8000;
			gamma_coef = 25;
		else
			C_lam	  = 2000;
			gamma_coef = 200;
		end
	else
		C_lam	  = 2400;
		gamma_coef = 50;
	end

	if sig > 0
		lam_start = 1e-2 * scA * (1 + C_lam * sig^2);
	else
		lam_start = 1e-2 * scA;
	end
	lam_min = 1e-6 * scA;
	lam	 = lam_start;

	geom_cool_rate = (lam_min / lam_start)^(1 / max(max_iters, 1));

	x_old = x;
	b_eff = b(:);

	if is_smooth
		eps_val = 1.0;
	else
		eps_val = 1.5e-3;
	end

	v = randn(N*N, 1);
	v = v / norm(v);
	for i = 1:5
		Av   = forward_op(v, ang, N, ndt);
		ATAv = adjoint_op(Av(:), ang, N, ndt);
		L_op = norm(ATAv(:));
		v	= ATAv(:) / max(L_op, 1e-20);
	end
	gamma_best = gamma_coef / max(L_op, 1);

	t = 1;

	for out_it = 1:max_iters

		if toc(t_st) > t_lim - 0.5
			break;
		end

		t_next = (1 + sqrt(1 + 4*t^2)) / 2;
		beta   = (t - 1) / t_next;

		y = x + beta * (x - x_old);
		y = max(xm, min(xM, y));

		[gx, gy]	= grad2D(reshape(y, N, N));
		grad_mag_sq = gx.^2 + gy.^2;

		edge_weight = 1 ./ (1 + grad_mag_sq / (delta^2));
		w		   = edge_weight ./ sqrt(grad_mag_sq + eps_val);

		x_prev = x;
		[x_new, r_norm, Ax_final] = cgls_inner( ...
			b_eff, b(:), y, w, lam, ang, N, ndt, 12);
		x = max(xm, min(xM, x_new));

		if sum((y - x) .* (x - x_prev)) > 0
			t = 1;
		else
			t = t_next;
		end

		x_old = x_prev;

		if ~is_smooth
			eps_val = max(3e-8, eps_val * 0.5);
		end

		if sig > 0
			dyn_rate = max(0.4, min(1.1, d_tgt / (r_norm + 1e-10)));
			lam	  = max(lam_min, lam * dyn_rate);
		else
			lam = max(lam_min, lam * geom_cool_rate);
		end

		if is_last && gamma_coef > 0
			b_eff = b_eff + gamma_best * (b(:) - Ax_final);
		end
	end
end

function x0 = fbp_init(b, angles, N, n_det)
	n_ang = length(angles);
	sino  = reshape(b, n_det, n_ang);

	npad  = 2^nextpow2(2 * n_det);

	freqs = (0 : npad-1)' / npad;
	freqs = freqs - (freqs > 0.5);

	ramp  = abs(freqs);
	hann  = 0.5 * (1 + cos(2*pi*freqs));
	filt  = ramp .* hann;

	sino_pad			 = zeros(npad, n_ang);
	sino_pad(1:n_det, :) = sino;

	S_sino	= custom_fft(sino_pad);
	S_filt	= bsxfun(@times, S_sino, filt);
	sino_filt = real(custom_ifft(S_filt));
	sino_filt = sino_filt(1:n_det, :);

	x0 = adjoint_op(sino_filt(:), angles, N, n_det);
	x0 = x0 * (pi / (2 * n_ang));
end

function X = custom_fft(x)
	N = size(x, 1);
	if N <= 1
		X = x;
		return;
	end
	even = custom_fft(x(1:2:end, :));
	odd  = custom_fft(x(2:2:end, :));
	
	factor = exp(-2i * pi * (0:N/2-1)' / N);
	factor_odd = bsxfun(@times, factor, odd);
	
	X = [even + factor_odd; even - factor_odd];
end

function x = custom_ifft(X)
	N = size(X, 1);
	x = conj(custom_fft(conj(X))) / N;
end

function x_up = upsample_image(x, N_old, N_new)
	img = reshape(x, N_old, N_old);

	[Xq, Yq] = meshgrid(linspace(1, N_old, N_new), ...
						linspace(1, N_old, N_new));

	img_up = interp2(img, Xq, Yq, 'linear', 0);
	x_up   = img_up(:);
end

function [x, r_norm, Ax] = cgls_inner(b_eff, b_true, x, w, lam, ang, N, ndt, max_it)
	Ax = forward_op(x, ang, N, ndt);
	r1 = b_eff(:) - Ax(:);

	[gx, gy] = grad2D(reshape(x, N, N));

	sq_lw = sqrt(lam .* w);

	r2x = -sq_lw .* gx;
	r2y = -sq_lw .* gy;

	div_res = div2D(sq_lw .* r2x, sq_lw .* r2y);

	d = adjoint_op(r1, ang, N, ndt) + div_res(:);
	d = d(:);
	p = d;

	norm_d_old = sum(d.^2);

	for i = 1:max_it
		q1 = forward_op(p, ang, N, ndt);

		[gpx, gpy] = grad2D(reshape(p, N, N));

		q2x = sq_lw .* gpx;
		q2y = sq_lw .* gpy;

		norm_q = sum(q1(:).^2) + sum(q2x(:).^2) + sum(q2y(:).^2);

		if norm_q < 1e-20
			break;
		end

		alpha = norm_d_old / norm_q;

		x   = x(:) + alpha * p;
		r1  = r1   - alpha * q1(:);
		r2x = r2x  - alpha * q2x;
		r2y = r2y  - alpha * q2y;

		div_res = div2D(sq_lw .* r2x, sq_lw .* r2y);

		d = adjoint_op(r1, ang, N, ndt) + div_res(:);
		d = d(:);

		norm_d_new = sum(d.^2);

		beta = norm_d_new / norm_d_old;

		p = d + beta * p;

		norm_d_old = norm_d_new;
	end

	Ax	 = b_eff(:) - r1;
	r_norm = norm(b_true(:) - Ax);
end

function [gx, gy] = grad2D(U)
	[M, N] = size(U);

	gx = zeros(M, N);
	gy = zeros(M, N);

	gx(1:end-1, :) = U(2:end,   :) - U(1:end-1, :);
	gy(:, 1:end-1) = U(:, 2:end)   - U(:, 1:end-1);
end

function D = div2D(px, py)
	[M, N] = size(px);

	dx = zeros(M, N);
	dy = zeros(M, N);

	dx(1,   :)	 = -px(1,   :);
	dx(2:end-1, :) =  px(1:end-2, :) - px(2:end-1, :);
	dx(end, :)	 =  px(end-1, :);

	dy(:,   1)	 = -py(:,   1);
	dy(:, 2:end-1) =  py(:, 1:end-2) - py(:, 2:end-1);
	dy(:, end)	 =  py(:, end-1);

	D = dx + dy;
end