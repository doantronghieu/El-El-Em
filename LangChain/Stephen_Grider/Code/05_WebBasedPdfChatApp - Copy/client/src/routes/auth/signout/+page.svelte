<script lang="ts">
	import { onMount } from 'svelte';
	import { goto, beforeNavigate } from '$app/navigation';
	import { signout, clearErrors } from '$s/auth';

	let timeout: number | null = null;

	onMount(async () => {
		await signout();

		timeout = setTimeout(() => {
			goto('/');
		}, 2500);
	});

	beforeNavigate(() => {
		clearErrors();
		if (timeout) {
			clearTimeout(timeout);
		}
	});
</script>

<main class="w-full max-w-md mx-auto p-6">
	<div
		class="mt-7 bg-white border border-gray-200 rounded-xl shadow-sm dark:bg-gray-800 dark:border-gray-700"
	>
		<div class="p-4 sm:p-7">
			<div class="text-center">
				Sad to see you go!
				<p>Redirecting...</p>
			</div>
		</div>
	</div>
</main>
